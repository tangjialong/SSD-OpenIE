from typing import Dict, List, TextIO, Optional, Any, Union
from overrides import overrides
import os
import torch
import torch.nn.functional as F
from torch.nn.modules import Linear, Dropout
from pytorch_transformers import BertForSequenceClassification, BertTokenizer
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
import requests

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure

def n_best_viterbi_decode(tag_sequence: torch.Tensor,
                          transition_matrix: torch.Tensor,
                          tag_observations: Optional[List[int]] = None,
                          n_best: int = 1):
    sequence_length, num_tags = list(tag_sequence.size())
    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ConfigurationError("Observations were provided, but they were not the same length "
                                     "as the sequence. Found sequence of length: {} and evidence: {}"
                                     .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    path_scores = []  # SHAPE: (seq_len * <=N * T)
    path_indices = []  # SHAPE: (seq_len * <=N * T)

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_sequence[:1, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        # SHAPE: (<=N, T, T)
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix.unsqueeze(0)
        # both SHAPE: (<=N, T)
        scores, paths = torch.topk(summed_potentials.view(-1, num_tags), n_best, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning("The pairwise potential between tags you have passed as "
                               "observations is extremely unlikely. Double check your evidence "
                               "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot.unsqueeze(0) + scores)
        else:
            path_scores.append(tag_sequence[timestep, :].unsqueeze(0) + scores)
        path_indices.append(paths)

    # Construct the top n most likely sequence backwards.
    # both SHAPE: (<=N)
    viterbi_score, ind = torch.topk(path_scores[-1].flatten(), n_best)
    ind_path = [ind]
    viterbi_path = [torch.remainder(ind, num_tags)]
    for backward_timestep in reversed(path_indices):
        ind = backward_timestep.flatten()[ind_path[-1]]
        ind_path.append(ind)
        viterbi_path.append(torch.remainder(ind, num_tags))
    # Reverse the backward path.
    viterbi_path.reverse()
    viterbi_path = torch.stack(viterbi_path, -1)
    return viterbi_path, viterbi_score

@Model.register("my-semisupervised-open-information-extraction")
class SemisupOpenIeModel(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 cuda_device: int = 0,
                 train_mode: str = "nll",
                 teacher_path: str = None,
                 beam_size: int = 3,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False) -> None:
        super(SemisupOpenIeModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")

        # For the span based evaluation, we don't want to consider labels
        # for verb, because the verb index is provided to the model.
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace="labels", ignore_classes=["V"])

        self.encoder = encoder
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes))
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric

        check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim,
                               encoder.get_input_dim(),
                               "text embedding dim + verb indicator embedding dim",
                               "encoder input dim") 

        if train_mode == "nll":
            self.label_weights = None # one can use label weighting as this: self.label_weights = {"O": 0.5}
        elif train_mode in ["rl_p", "rl_h"]:
            pass
        elif train_mode in ["rl_t", "rl_pt"]:
            self.tokenizer = BertTokenizer.from_pretrained(teacher_path, do_lower_case=True)
            self.teacher_model = BertForSequenceClassification.from_pretrained(teacher_path, num_labels=2)
            self.teacher_model.to(cuda_device)
            for param in self.teacher_model.parameters():
                param.requires_grad = False # do not update the teacher BERT
        else:
            raise ConfigurationError(f"{train_mode} not supported!")

        self.cuda_device = cuda_device
        self.train_mode = train_mode
        self.beam_size = beam_size
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                head_indicator: torch.LongTensor = None,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(verb_indicator.long())
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, embedded_verb_indicator], -1)
        batch_size, sequence_length, _ = embedded_text_with_verb_indicator.size()

        encoded_text = self.encoder(embedded_text_with_verb_indicator, mask)

        logits = self.tag_projection_layer(encoded_text)
        
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                        sequence_length,
                                                                        self.num_classes])

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask

        words, verbs = zip(*[(x["words"], x["verb"]) for x in metadata])
        if metadata is not None:
            output_dict["words"] = list(words)
            output_dict["verb"] = list(verbs)
        
        if tags is not None:
            if self.train_mode == "nll":
                loss = self.get_cross_entropy_loss(logits, tags, mask, label_weights=self.label_weights, head_indicator=None)
            elif self.train_mode in ["rl_p", "rl_t", "rl_pt", "rl_h"]:
                loss = self.get_RL_sen_reward_loss(logits, class_probabilities, tags, mask, head_indicator=head_indicator, orign_sen=output_dict["words"])
            else:
                ConfigurationError(f"{train_mode} not supported!")
            # debug
            # self.show_grad(loss)
            output_dict["loss"] = loss
            if not self.ignore_span_metric:
                self.span_metric(class_probabilities, tags, mask)

        return output_dict

    def get_cross_entropy_loss(self, logits, targets, weights, label_weights=None, head_indicator=None):
        # shape : (batch * sequence_length, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
        # shape : (batch * max_len, 1)
        targets_flat = targets.view(-1, 1).long()

        if label_weights is not None:
            # label weighting
            batch_size, sequence_length = targets.size()
            label_weights_vector = torch.ones(self.num_classes)
            for label, label_weight in label_weights.items():
                label_weights_vector[self.vocab.get_token_index(label, namespace="labels")] = label_weight
            label_weights_vector = label_weights_vector.expand(batch_size*sequence_length, self.num_classes)
            log_probs_flat = log_probs_flat * label_weights_vector.cuda(self.cuda_device)

        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())

        if head_indicator is not None:
            # head word is more important
            head_word_weights = head_indicator + 1.
            negative_log_likelihood = negative_log_likelihood * head_word_weights.float() 

        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood * weights.float()

        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)

        return per_batch_loss.sum() / num_non_empty_sequences

    def get_RL_sen_reward_loss(self, logits, class_probabilities, tags, mask, head_indicator=None, orign_sen=None):
        batch_size, sequence_length = tags.size()
        # get class probailities, shape (batch_size, sequence_length, num_classes)
        all_class_probabilities = class_probabilities.detach().cpu().numpy()
        # beam search to get actions, shape: (beam_size, batch_size, sequence_length)
        actions = torch.zeros(self.beam_size, batch_size, sequence_length)
        i = 0
        for this_class_probabilities, this_masks in zip(all_class_probabilities, mask.cpu().numpy()):
            all_tags, all_probs = self._beam_search(this_class_probabilities, this_masks, n_best=self.beam_size)
            for z in range(self.beam_size):
                for j in range(len(all_tags[z])):
                    actions[z][i][j] = self.vocab.get_token_index(all_tags[z][j], namespace="labels")
            i = i + 1
        
        # debug
        # self.show_beamsearch(actions, tags.cpu().numpy(), mask.cpu().numpy(), orign_sen)

        # get losses
        losses = []

        for action in actions:
            action = action.cuda(self.cuda_device).long()
            action_neg_log_probabilities = - torch.gather(torch.log(class_probabilities + 1e-13), dim=2, index=action.view(batch_size, sequence_length, 1)).view(batch_size, sequence_length)
            action_neg_log_probabilities = action_neg_log_probabilities * mask.float()
            # shape : (batch_size,)
            avg_neg_log_probs = action_neg_log_probabilities.sum(1) / (mask.sum(1).float() + 1e-13)
            # get reward and instances weights
            if self.train_mode == "rl_h":
                half_alp = avg_neg_log_probs / 2 
                reward, instances_weights = self._get_reward_H(action, class_probabilities, mask, head_indicator, batch_size, sequence_length)
                loss1 = F.margin_ranking_loss(-half_alp, half_alp, reward.float(), margin=1.0, reduction='none') * instances_weights
            else:
                if self.train_mode == "rl_p":
                    reward = self._get_reward_P(action, head_indicator, batch_size, sequence_length)
                elif self.train_mode == "rl_t":
                    reward = self._get_reward_T(action, orign_sen, batch_size, sequence_length)
                elif self.train_mode == "rl_pt":
                    reward = self._get_reward_PT(action, orign_sen, head_indicator, batch_size, sequence_length)
                else:
                    ConfigurationError(f"{train_mode} not supported!")
                loss1 = avg_neg_log_probs * reward
            losses.append(loss1)

        num_non_empty_sequences = ((mask.sum(1) > 0).float().sum() + 1e-13)
        loss = torch.zeros(batch_size).cuda(self.cuda_device)
        for loss1 in losses:
            loss = loss + loss1
        final_loss = loss.sum() / num_non_empty_sequences / self.beam_size
        return final_loss

    def _beam_search(self, prob: np.ndarray, mask: np.ndarray, n_best: int = 1):
        log_prob = np.log(prob + 1e-13)
        seq_lens = mask.sum(-1).tolist()
        one_sam = False
        if log_prob.ndim == 2:
            one_sam = True
            p_li, lp_li, seq_lens = [prob], [log_prob], [seq_lens]
        else:
            p_li, lp_li, seq_lens = prob, log_prob, seq_lens
        all_tags, all_probs = [], []
        trans_mat = self.get_viterbi_pairwise_potentials()
        for p, lp, slen in zip(p_li, lp_li, seq_lens):
            # viterbi decoding (based on torch tensor)
            vpaths, vscores = n_best_viterbi_decode(
                torch.from_numpy(lp[:slen]), trans_mat, n_best=n_best)
            vpaths = vpaths.numpy()
            # collect tags and corresponding probs
            cur_tags, cur_probs = [], []
            for vpath in vpaths:
                probs = [p[i, vpath[i]] for i in range(len(vpath))]
                tags = [self.vocab.get_token_from_index(x, namespace='labels') for x in vpath]
                cur_probs.append(probs)
                cur_tags.append(tags)
            all_probs.append(cur_probs)
            all_tags.append(cur_tags)
        if one_sam:
            return all_tags[0], all_probs[0]
        return all_tags, all_probs

    def _get_reward_H(self, action, class_probabilities, mask, head_indicator, batch_size, sequence_length):
        all_tags = action.cpu().numpy()
        if head_indicator is not None:
            all_heads = head_indicator.cpu().numpy()
        else:
            all_heads = torch.zeros(batch_size, sequence_length).numpy()
        
        reward = []
        for tags_a, heads_a in zip(all_tags, all_heads):
            flag = 1
            arg_head_num = 0
            arg_num = 0
            for tag_a, head_a in zip(tags_a, heads_a):
                if head_a == 2:
                    arg_head_num += 1
                if "B-ARG" in self.vocab.get_token_from_index(tag_a, namespace="labels"):
                    arg_num += 1
                if head_a == 1 and '-V' not in self.vocab.get_token_from_index(tag_a, namespace="labels"):
                    flag = -1
                elif head_a == 2 and '-ARG' not in self.vocab.get_token_from_index(tag_a, namespace="labels"):
                    flag = -1
            if arg_head_num != arg_num:
                flag = -1
            reward.append(flag)
        
        reward = torch.FloatTensor(reward).cuda(self.cuda_device)
        instances_weights = torch.gather(class_probabilities, dim=2, index=action.view(batch_size, sequence_length, 1)).view(batch_size, sequence_length) * mask.float()
        instances_weights = instances_weights.sum(1) / (mask.sum(1).float() + 1e-13)
        return reward, instances_weights.detach() # do not need backward

    def _get_reward_P(self, action, head_indicator, batch_size, sequence_length):
        all_tags = action.cpu().numpy()
        if head_indicator is not None:
            all_heads = head_indicator.cpu().numpy()
        else:
            all_heads = torch.zeros(batch_size, sequence_length).numpy()
        
        reward = []
        for tags_a, heads_a in zip(all_tags, all_heads):
            flag = 1
            arg_head_num = 0
            arg_num = 0
            for tag_a, head_a in zip(tags_a, heads_a):
                if head_a == 2:
                    arg_head_num += 1
                if "B-ARG" in self.vocab.get_token_from_index(tag_a, namespace="labels"):
                    arg_num += 1
                if head_a == 1 and '-V' not in self.vocab.get_token_from_index(tag_a, namespace="labels"):
                    flag = -1
                elif head_a == 2 and '-ARG' not in self.vocab.get_token_from_index(tag_a, namespace="labels"):
                    flag = -1
            if arg_head_num != arg_num:
                flag = -1
            reward.append(flag)
        
        reward = torch.FloatTensor(reward).cuda(self.cuda_device)
        return reward

    def _get_reward_T(self, action, orign_sen, batch_size, sequence_length):
        all_tags = action.cpu().numpy()

        tokens_a = orign_sen
        tokens_b = [[] for i in range(batch_size)]
        tokens_ab = [[] for i in range(batch_size)]
        input_ids = [[] for i in range(batch_size)]
        segment_ids = [[] for i in range(batch_size)]
        input_mask = [[] for i in range(batch_size)]

        for i in range (batch_size):
            for j in range(sequence_length):
                if j < len(tokens_a[i]) and self.vocab.get_token_from_index(all_tags[i][j], namespace="labels") != "O":
                    if '-V' in self.vocab.get_token_from_index(all_tags[i][j], namespace="labels"):
                        tokens_b[i].append('@@ ' + tokens_a[i][j])
                    else:
                        tokens_b[i].append(tokens_a[i][j])
            tokens_ab[i], input_ids[i], segment_ids[i], input_mask[i] = self._convert_list_to_teacherinput(tokens_a[i], tokens_b[i])
        
        input_ids, input_mask, segment_ids = self._padding(input_ids, input_mask, segment_ids)
        input_ids = torch.LongTensor(input_ids).cuda(self.cuda_device)
        input_mask = torch.LongTensor(input_mask).cuda(self.cuda_device)
        segment_ids = torch.LongTensor(segment_ids).cuda(self.cuda_device)

        # shape: (batch_size * num_label)
        r = self.teacher_model(input_ids = input_ids,
                               attention_mask = input_mask,
                               token_type_ids = segment_ids)[0].cpu().numpy()
        
        def softmax(z):
	        return np.exp(z)/sum(np.exp(z))
        reward = []
        for this_r in r:
            reward.append(softmax(this_r)[1])
        reward = torch.FloatTensor(reward).cuda(self.cuda_device)

        return reward

    def _get_reward_PT(self, action, orign_sen, head_indicator, batch_size, sequence_length):
        all_tags = action.cpu().numpy()
        if head_indicator is not None:
            all_heads = head_indicator.cpu().numpy()
        else:
            all_heads = torch.zeros(batch_size, sequence_length).numpy()
        
        reward1 = []
        for tags_a, heads_a in zip(all_tags, all_heads):
            flag = 1
            arg_head_num = 0
            arg_num = 0
            for tag_a, head_a in zip(tags_a, heads_a):
                if head_a == 2:
                    arg_head_num += 1
                if "B-ARG" in self.vocab.get_token_from_index(tag_a, namespace="labels"):
                    arg_num += 1
                if head_a == 1 and '-V' not in self.vocab.get_token_from_index(tag_a, namespace="labels"):
                    flag = -1
                elif head_a == 2 and '-ARG' not in self.vocab.get_token_from_index(tag_a, namespace="labels"):
                    flag = -1
            if arg_head_num != arg_num:
                flag = -1
            reward1.append(flag)
        
        reward1 = torch.FloatTensor(reward1).cuda(self.cuda_device)

        tokens_a = orign_sen
        tokens_b = [[] for i in range(batch_size)]
        tokens_ab = [[] for i in range(batch_size)]
        input_ids = [[] for i in range(batch_size)]
        segment_ids = [[] for i in range(batch_size)]
        input_mask = [[] for i in range(batch_size)]

        for i in range (batch_size):
            for j in range(sequence_length):
                if j < len(tokens_a[i]) and self.vocab.get_token_from_index(all_tags[i][j], namespace="labels") != "O":
                    if '-V' in self.vocab.get_token_from_index(all_tags[i][j], namespace="labels"):
                        tokens_b[i].append('@@ ' + tokens_a[i][j])
                    else:
                        tokens_b[i].append(tokens_a[i][j])
            tokens_ab[i], input_ids[i], segment_ids[i], input_mask[i] = self._convert_list_to_teacherinput(tokens_a[i], tokens_b[i])
        
        input_ids, input_mask, segment_ids = self._padding(input_ids, input_mask, segment_ids)
        input_ids = torch.LongTensor(input_ids).cuda(self.cuda_device)
        input_mask = torch.LongTensor(input_mask).cuda(self.cuda_device)
        segment_ids = torch.LongTensor(segment_ids).cuda(self.cuda_device)

        # shape: (batch_size * num_label)
        r = self.teacher_model(input_ids = input_ids,
                               attention_mask = input_mask,
                               token_type_ids = segment_ids)[0].cpu().numpy()
        
        def softmax(z):
	        return np.exp(z)/sum(np.exp(z))
        reward2 = []
        for this_r in r:
            reward2.append(softmax(this_r)[1])
        reward2 = torch.FloatTensor(reward2).cuda(self.cuda_device)

        return reward1 * reward2.detach() # do not need backward

    def _convert_list_to_teacherinput(self, tokens_a, tokens_b):
        tmp_a = self.tokenizer.tokenize(' '.join(tokens_a))
        tmp_b = self.tokenizer.tokenize(' '.join(tokens_b)) 
        tokens_ab = ["[CLS]"] + tmp_a + ["[SEP]"] + tmp_b + ["[SEP]"]
        segment_ids = [0] * (len(tmp_a) + 2) + [1] * (len(tmp_b) + 1)
        # get id and mask
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_ab)
        input_mask = [1] * len(input_ids)
        return tokens_ab, input_ids, segment_ids, input_mask
    
    def _padding(self, input_ids, input_mask, segment_ids):
        # padding to max sequence length
        max_sequence_length = 0
        for i in range(len(input_ids)):
            if len(input_ids[i]) > max_sequence_length:
                max_sequence_length = len(input_ids[i])
        for i in range(len(input_ids)):
            padding = [0] * (max_sequence_length - len(input_ids[i]))
            input_ids[i] += padding
            input_mask[i] += padding
            segment_ids[i] += padding
        return input_ids, input_mask, segment_ids

    # used to debug
    def show_grad(self, loss):
        loss.backward()
        print ("$$$$$$$$$$$$$$$$$$$")
        for name, params in self.named_parameters():
            print ("----------------")
            print (name)
            if params.requires_grad:
                print (params.grad)
            else:
                print ("Frozen")
        print ("----------------")
        print (loss)
        quit()

    # used to debug
    def show_beamsearch(self, actions, tags, mask, orign_sen):
        beam_size, batch_size, sequence_length = actions.size()
        for i in range(batch_size):
            print ("$$$$$$$$$$$$$$$$$$$")
            print (' '.join(orign_sen[i]))
            print ("----------------")
            outinfo = ""
            for j in range(sequence_length):
                if mask[i][j] == 0.:
                    break
                outinfo = outinfo + '%7s'%(self.vocab.get_token_from_index(tags[i][j], namespace="labels")) + ' '
            print (outinfo)

            for k in range(beam_size):
                print ("----------------")
                outinfo = ""
                for j in range(sequence_length):
                    if mask[i][j] == 0.:
                        break
                    outinfo = outinfo + '%7s'%self.vocab.get_token_from_index(int(actions[k][i][j]), namespace="labels") + ' '
                print (outinfo)
        quit()

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the
            # span metric
            return {}

        else:
            metric_dict = self.span_metric.get_metric(reset=reset)

            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "overall" in x}

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

def write_to_conll_eval_file(prediction_file: TextIO,
                             gold_file: TextIO,
                             verb_index: Optional[int],
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str]):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    verb_only_sentence = ["-"] * len(sentence)
    if verb_index:
        verb_only_sentence[verb_index] = sentence[verb_index]

    conll_format_predictions = convert_bio_tags_to_conll_format(prediction)
    conll_format_gold_labels = convert_bio_tags_to_conll_format(gold_labels)

    for word, predicted, gold in zip(verb_only_sentence,
                                     conll_format_predictions,
                                     conll_format_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")

def convert_bio_tags_to_conll_format(labels: List[str]):
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").

    A full example of the conversion performed:

    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]

    Parameters
    ----------
    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.

    Returns
    -------
    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels
