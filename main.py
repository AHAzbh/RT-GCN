import argparse
import os
import random
import time
import logging

from graph import NodeGraph
import utils
from dataset import REProcessor
from gcn import GCN

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_scheduler, BertModel, BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              WeightedRandomSampler, TensorDataset, Dataset)
from tqdm.auto import tqdm

from loss import FocalLoss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    filename='./log/new-{}.log'.format(time.strftime("%Y%m%d%H%M", time.localtime())),
                    filemode='w+',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default="./data/ace",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default="gcn-01",
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="./output",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_path",
                        default="./model",
                        type=str,
                        required=False,
                        help="Model path")
    parser.add_argument("--model_name",
                        default="rt-gcn",
                        type=str,
                        help="Model name")
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
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to predict.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--reconstruct_graph",
                        action='store_true',
                        help="Set this flag if you need to reconstruct graph.")
    parser.add_argument("--weighted_random_sampler",
                        action='store_true',
                        help="Set this flag if you need to reconstruct graph.")
    parser.add_argument("--plm",
                        default="bert-base-cased",
                        type=str,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--num_workers",
                        default=0,
                        type=int,
                        help="num workers for training.")
    parser.add_argument("--num_samples",
                        default=50000,
                        type=int,
                        help="num samples for training.")
    parser.add_argument("--link_mode",
                        default=1,
                        type=int,
                        help="link mode.")
    parser.add_argument("--train_batch_size",
                        default=10,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--evaluate_after_steps",
                        default=100,
                        type=int,
                        help="evaluate after steps.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gcn_config_index",
                        default=0,
                        type=int,
                        help="gcn config index.")
    parser.add_argument("--normal_config_index",
                        default=0,
                        type=int,
                        help="normal config index.")
    parser.add_argument("--hidden_dropout_prob",
                        default=0.4,
                        type=float,
                        help="hidden_dropout_prob.")
    parser.add_argument("--num_attention_heads",
                        default=10,
                        type=int,
                        help="num_attention_heads.")
    parser.add_argument("--max_steps",
                        default=-1,
                        type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--focal_loss",
                        action='store_true',
                        help="Set this flag if you need to reconstruct graph.")
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
                        default=1,
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
    parser.add_argument('--vocab_file',
                        type=str, default=None,
                        help="Vocabulary mapping/file BERT was pretrainined on")

    parser.add_argument('--num_gcn_layers', type=int, default=2)

    args = parser.parse_args()

    args.task_name = args.task_name.lower()

    if args.gcn_config_index == 1:
        args.gcn_features = [2000, 100, 5]
    elif args.gcn_config_index == 2:
        args.gcn_features = [800, 2000, 500, 1200, 200, 5]
    else:
        args.gcn_features = [2000, 500, 200, 50, 10, 5]

    if args.normal_config_index == 1:
        args.normal_features = [500]
    elif args.normal_config_index == 2:
        args.normal_features = [800, 1500, 500]
    else:
        args.normal_features = [2000, 500]

    return args


def train(args, model, label_index, processor, results=None):
    if results is None:
        results = {}
    results["best_checkpoint"] = 0
    results["best_micro_f1_score"] = 0
    results["best_macro_f1_score"] = 0
    results["best_dev_f1_score"] = 0
    results["best_mrr_score"] = 0
    results["best_checkpoint_path"] = ""
    results["best_micro_f1_step"], results["best_macro_f1_step"] = 0, 0

    print(args.data_dir)

    train_examples = processor.get_train_examples()
    count_index = dict()
    for _, _, _, _, label in train_examples:
        if label in count_index:
            count_index[label] += 1
        else:
            count_index[label] = 1

    if args.weighted_random_sampler:
        weights = [len(train_examples) / count_index[label] for _, _, _, _, label in train_examples]
        sampler = WeightedRandomSampler(weights=weights, num_samples=args.num_samples, replacement=True)
        dataloader = DataLoader(train_examples, sampler=sampler, batch_size=args.train_batch_size,
                                num_workers=args.num_workers, drop_last=True)
        if args.focal_loss:
            loss_func = FocalLoss(class_num=len(label_index))
        else:
            loss_func = CrossEntropyLoss()
    else:
        dataloader = DataLoader(train_examples, batch_size=args.train_batch_size,
                                num_workers=args.num_workers, shuffle=True, drop_last=True)
        loss_weights = torch.tensor([len(train_examples) / count_index[label] / 10 for label in range(len(label_index))])
        if args.focal_loss:
            loss_func = FocalLoss(class_num=len(label_index), alpha=loss_weights)
        else:
            loss_func = CrossEntropyLoss(weight=loss_weights).to(args.device)

    # dataloader = DataLoader(train_examples, sampler=sampler, batch_size=args.train_batch_size,
    #                         num_workers=args.num_workers, drop_last=True)

    num_training_steps = args.num_train_epochs * len(dataloader)

    num_train_optimization_steps = int(
        len(dataloader) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    print("lr: {} warm: {} total_step: {}".format(args.learning_rate, args.warmup_proportion,
                                                  num_training_steps))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_optimization_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    logger.info("model config : batch_size={}, gradient_accumulation_steps={}, lr={}, link_mode={}, "
                "dropout_prob={}, gcn_feature={}, normal_feature={}, epochs={}".
                format(args.train_batch_size, args.gradient_accumulation_steps, args.learning_rate,
                       args.link_mode, args.hidden_dropout_prob, str(args.gcn_features),
                       str(args.normal_features), args.num_train_epochs))

    model.train()
    for epoch in range(args.num_train_epochs):
        for batch in dataloader:
            texts, label_masks, sub_types, obj_types, labels = batch

            try:
                # torch.cuda.empty_cache()
                logits = model(texts, label_masks, sub_types, obj_types)
                # print(logits.device)
                # labels = torch.tensor(labels).to(args.device).long()
                # print(logits.device, labels.device)
                # print(logits)
                loss = loss_func(logits, labels.to(args.device).long())
                # print(loss)

            except Exception as e:
                logger.info("after {} steps, batch is {}, error is {}".format(progress_bar.n, batch, e))
                raise ValueError("loss error")

            if torch.isnan(loss):
                logger.info("after {} steps, loss is nan, {} trained".
                            format(progress_bar.n, progress_bar.n / num_training_steps))
                raise ValueError("steps {}, loss is nan".format(progress_bar.n))

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            progress_bar.update(1)

            if progress_bar.n % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if progress_bar.n % args.evaluate_after_steps == 0:
                result = evaluate(args, model, label_index, processor)
                better_micro, better_macro = False, False
                if result["micro_f1"] > results["best_micro_f1_score"]:
                    better_micro = True
                    results["best_micro_f1_score"] = result["micro_f1"]
                    results["best_micro_f1_step"] = progress_bar.n
                    output_dir = os.path.join(args.output_dir, "best_micro_f1")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    utils.save_zen_model(output_dir, model, model.tokenizer)

                if result["macro_f1"] > results["best_macro_f1_score"]:
                    better_macro = True
                    results["best_macro_f1_score"] = result["macro_f1"]
                    results["best_macro_f1_step"] = progress_bar.n
                    output_dir = os.path.join(args.output_dir, "best_macro_f1")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    utils.save_zen_model(output_dir, model, model.tokenizer)
                logger.info("after {} steps , better micro {}, better macro {}, result is {}, loss is {}"
                            .format(progress_bar.n, better_micro, better_macro, result, loss))

    logger.info("train results : {}".format(results))


def evaluate(args, model, label_index, processor, mode="dev"):
    if mode == "test":
        examples = processor.get_test_examples()
    elif mode == "dev":
        examples = processor.get_dev_examples()
    eval_sampler = SequentialSampler(examples)
    eval_dataloader = DataLoader(examples, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    all_preds, all_labels = [], []
    eval_start_time = time.time()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        texts, label_masks, sub_types, obj_types, labels = batch

        with torch.no_grad():
            # torch.cuda.empty_cache()
            logits = model(texts, label_masks, sub_types, obj_types)

        preds = torch.argmax(logits, dim=-1)
        all_preds = np.append(all_preds, preds.detach().cpu().numpy(), axis=0)
        all_labels = np.append(all_labels, labels.detach().cpu().numpy(), axis=0)

    micro_f1, macro_f1 = utils.cal_f1(all_preds, all_labels, label_index, ignore_label="Other")
    eval_run_time = time.time() - eval_start_time

    # if args.task_name == 'semeval':
    #     result = semeval_official_eval(id2label_map, preds, out_label_ids, output_dir)
    # else:
    #     result = {
    #         "f1": compute_micro_f1(preds, out_label_ids, label_map, ignore_label='Other', output_dir=output_dir)
    #     }
    result = dict()
    result["micro_f1"] = micro_f1
    result["macro_f1"] = macro_f1
    result["eval_run_time"] = eval_run_time
    result["inference_time"] = eval_run_time / len(examples)
    return result


def train_func(args):
    args.device = device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    args.n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and utils.is_main_process():
        os.makedirs(args.output_dir)

    plm = BertModel.from_pretrained(args.plm).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.plm)

    if args.reconstruct_graph:
        graph = NodeGraph(args.data_dir)
        utils.store_graph(graph)
    graph = utils.load_graph(args.data_dir)
    model = GCN(args.data_dir, graph=graph, gcn_features=args.gcn_features, normal_features=args.normal_features,
                link_mode=args.link_mode, plm=plm, tokenizer=tokenizer, batch_size=args.train_batch_size,
                hidden_dropout_prob=args.hidden_dropout_prob, num_attention_heads=args.num_attention_heads, device=device)
    node_index, label_index = graph.node_index, graph.label_index

    processor = REProcessor(args.data_dir, node_index, label_index, tokenizer, device)

    model = model.to(device)
    print("build GCN done")

    # print("parameters device {}".format(next(model.parameters()).device))
    # for name, para in model.named_parameters():
    #     if para.requires_grad:
    #         print(name, type(para), para.size(), para.device)
    # print("model to device {}".format(device))

    train(args, model, label_index, processor)


def test_func(args):
    args.device = device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    print("LOAD CHECKPOINT from", args.model_path)
    model = GCN.from_pretrained(args.model_path)
    # dict_bin = torch.load(os.path.join(args.model_path, "dict.bin"))

    tokenizer = BertTokenizer.from_pretrained(args.plm)

    graph = utils.load_graph()
    label_index, node_index = graph.label_index, graph.node_index
    processor = REProcessor(args.data_dir, node_index, label_index, tokenizer, device)
    model = model.to(device)
    result = evaluate(args, model, label_index, processor, mode="test")
    print(result)


def predict_func(args):
    pass


def main():
    args = get_args()
    if args.do_train:
        train_func(args)
    elif args.do_test:
        test_func(args)
    elif args.do_predict:
        predict_func(args)


if __name__ == "__main__":
    main()
