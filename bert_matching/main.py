import logging
import os

import numpy as np
import torch
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

from base.bert_utils import add_parser_arguments, prepare_optimizer
from base.bert_utils import p_r_f1, log_format
from base.bert_utils import prepare_model, prepare_env
from base.bert_utils import save_model, load_model
from base.data_loader import get_data_loader, get_processor
from base.io_utils import convert_examples_to_features

logger = logging.getLogger(__name__)


def compute_metrics(preds, labels, average='binary'):
    assert len(preds) == len(labels)
    return p_r_f1(preds, labels, average)


def eval_model(model, eval_data_loader, label_list, args, device, prefix=""):
    num_labels = len(label_list)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data_loader.dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    labels = []

    eval_average = 'binary' if args.processor == 'matching' else 'micro'

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_data_loader, desc="Evaluating"):
        labels += [label_ids]

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            if isinstance(logits, tuple):
                logits = logits[0]

        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(
            logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    store_preds = preds ##############
    preds = np.argmax(preds, axis=1)
    labels = torch.cat(labels)
    result = compute_metrics(preds, labels.numpy(), eval_average)

    result['eval_loss'] = eval_loss

    output_eval_file = os.path.join(
        args.output_dir, prefix + "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if prefix == 'final_test':    
        with open(args.data_dir + "/pred.tsv", "w") as writer:
            with open(args.data_dir + "/test.tsv") as reader:
                for p, l, line, s_p in zip(preds, labels.numpy(), reader, store_preds):
                    line = line.strip().split('\t')
                    if line[2]!= str(l):
                        print ("ERROR!!!")
                        quit()
                    writer.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (line[0], line[1], line[2], str(p), str(s_p[0]), str(s_p[1])))

    return result


def train_model(model,
                tokenizer,
                train_data_loader,
                eval_data_loader,
                label_list,
                args,
                optimizer,
                warmup_linear,
                env_option
                ):
    device = env_option['device']
    n_gpu = env_option['n_gpu']
    num_train_optimization_steps = env_option['num_train_optimization_steps']

    num_labels = len(label_list)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_loader.dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    global_step = 0
    best_score = 0.
    early_stop = 0
    for epoch_index in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step_index, batch in enumerate(tqdm(train_data_loader, desc="Step")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            if isinstance(logits, tuple):
                logits = logits[0]

            # label weighting
            loss_fct = CrossEntropyLoss(weight=torch.tensor([1., 5.]).to(device))

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step_index + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * \
                                   warmup_linear.get_lr(
                                       global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if args.eval_each_step > 0 and (step_index + 1) % args.eval_each_step == 0:
                valid_result = eval_model(model, eval_data_loader, label_list, args, device,
                                          prefix="epoch_%d_dev" % epoch_index)
                logger.info(log_format(
                    valid_result, prefix="Valid at Step %s" % step_index))
                if valid_result['score'] > best_score:
                    save_model(model, tokenizer=tokenizer, args=args)
                    best_score = valid_result['score']
                    early_stop = 0
                else:
                    early_stop += 1
                logger.info("early_stop is %d" % early_stop)
                if early_stop >= 5:
                    return 

        valid_result = eval_model(model, eval_data_loader, label_list, args, device,
                                  prefix="epoch_%d_dev" % epoch_index)
        logger.info(log_format(
            valid_result, prefix="Valid at Epoch %s" % epoch_index))
        if valid_result['score'] > best_score:
            save_model(model, tokenizer=tokenizer, args=args)
            best_score = valid_result['score']
            early_stop = 0
        else:
            early_stop += 1
        logger.info("early_stop is %d" % early_stop)
        if early_stop >= 10:
            return 


def main():
    args = add_parser_arguments()

    env_option = prepare_env(args)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = get_processor(args.processor)

    label_list = processor.get_labels()
    num_labels = len(label_list)

    logger.info("Load Bert ...")
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    logger.info(args.bert_model)
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.bert_model,
                                                          num_labels=num_labels)
    model = prepare_model(model, args, env_option)

    if args.do_train:
        valid_features_to_bert = convert_examples_to_features(examples=processor.get_dev_examples(args.data_dir),
                                                            label_list=label_list,
                                                            max_seq_length=args.max_seq_length,
                                                            tokenizer=tokenizer,
                                                            output_mode="classification")
        valid_data_loader = get_data_loader(
            valid_features_to_bert, args, is_train=False)

        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        env_option['num_train_optimization_steps'] = num_train_optimization_steps

        model, optimizer, warmup_linear = prepare_optimizer(
            model, args, env_option)

        train_features_to_bert = convert_examples_to_features(examples=train_examples,
                                                              label_list=label_list,
                                                              max_seq_length=args.max_seq_length,
                                                              tokenizer=tokenizer,
                                                              output_mode="classification")
        train_data_loader = get_data_loader(
            train_features_to_bert, args, is_train=True)
        train_model(model=model,
                    tokenizer=tokenizer,
                    train_data_loader=train_data_loader,
                    eval_data_loader=valid_data_loader,
                    label_list=label_list,
                    args=args,
                    optimizer=optimizer,
                    warmup_linear=warmup_linear,
                    env_option=env_option)


    if args.do_eval or args.do_train:
        model, tokenizer = load_model(
            args=args, device=env_option['device'], num_labels=num_labels)

        test_features_to_bert = convert_examples_to_features(examples=processor.get_test_examples(args.data_dir),
                                                             label_list=label_list,
                                                             max_seq_length=args.max_seq_length,
                                                             tokenizer=tokenizer,
                                                             output_mode="classification")
        test_data_loader = get_data_loader(
            test_features_to_bert, args, is_train=False)
        test_result = eval_model(model=model,
                                 eval_data_loader=test_data_loader,
                                 label_list=label_list,
                                 args=args,
                                 device=env_option['device'],
                                 prefix="final_test",
                                 )
        print(test_result)


if __name__ == "__main__":
    main()
