import argparse
import logging
import os
import random
import numpy as np
import torch
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertForSequenceClassification, BertTokenizer
from pytorch_transformers.optimization import WarmupLinearSchedule, AdamW

logger = logging.getLogger(__name__)


def add_parser_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--processor",
                        default='matching',
                        type=str,
                        required=True,
                        help="Processor for Data folder, matching, snli")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input zero_shot_data dir. Should contain the .tsv files (or other zero_shot_data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--device",
                        default=None,
                        type=str,
                        help="Device Str")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_each_step",
                        default=100,
                        type=int,
                        help="Eval model for each n steps.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")
    args = parser.parse_args()
    if "uncased" in args.bert_model:
        assert args.do_lower_case is True
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print("Set CUDA_VISIBLE_DEVICES as %s" % args.device)
    return args


def prepare_optimizer(model, args, env_option):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                     t_total=env_option['num_train_optimization_steps'])

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device.split(',')) > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    return model, optimizer, scheduler


def save_model(model, tokenizer, args):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)


def load_model(args, device, num_labels):
    logger.info("Load model from %s" % args.output_dir)
    model = BertForSequenceClassification.from_pretrained(
        args.output_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case)
    model.to(device)
    return model, tokenizer


def prepare_env(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    env_option = {'device': device,
                  'n_gpu': n_gpu}

    return env_option


def prepare_model(model, args, env_option):
    if args.fp16:
        model.half()
    model.to(env_option['device'])
    return model


def p_r_f1(preds, labels, average='binary'):
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.metrics import accuracy_score
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    precision = precision_score(y_true=labels, y_pred=preds, average=average)
    recall = recall_score(y_true=labels, y_pred=preds, average=average)
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {
        "p": precision,
        "r": recall,
        "f1": f1,
        'acc': acc,
        'score': f1
    }


def log_format(result, prefix=None):
    report_list = list()
    if 'eval_loss' in result:
        report_list += ["Loss: %.4f" % result['eval_loss']]
    if 'acc' in result:
        report_list += ["Acc: %.2f" % (result['acc'] * 100)]
    if 'p' in result:
        report_list += ["P: %.2f" % (result['p'] * 100)]
    if 'r' in result:
        report_list += ["R: %.2f" % (result['r'] * 100)]
    if 'f1' in result:
        report_list += ["F1: %.2f" % (result['f1'] * 100)]
    result_str = ', '.join(report_list)
    if prefix:
        return "%s %s" % (prefix, result_str)
    else:
        return result_str
