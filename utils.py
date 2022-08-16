import os
import pickle

import torch
from transformers import WEIGHTS_NAME, CONFIG_NAME
import torch.distributed as dist
from sklearn.metrics import f1_score, classification_report


def cal_f1(preds, labels, label_map, ignore_label):
    target_name = []
    target_id = []
    for name, id in label_map.items():
        if name in ignore_label:
            continue
        target_id.append(id)
        target_name.append(name)
    res = classification_report(labels, preds, labels=target_id, target_names=target_name, output_dict=True)
    if "micro avg" in res:
        return res["micro avg"]["f1-score"], res["macro avg"]["f1-score"]
    else:
        return res["accuracy"], res["macro avg"]["f1-score"]


def get_label_mask(tokens, token_mask, tokenizer):
    start, end, ind = 0, 0, 0
    mask_map = dict()
    for mask in token_mask:
        mask_map[mask[0]] = ""
        mask_map[mask[1]] = ""

    for token in tokens:
        start = end
        end += len(tokenizer.tokenize(token))
        if ind in mask_map:
            mask_map[ind] = (start, end)
        ind += 1
    mask_map[ind] = (end, end)

    label_mask = list()
    for mask in token_mask:
        label_mask.append((mask_map[mask[0]][0], mask_map[mask[1]][0]))

    return label_mask


def save_zen_model(save_zen_model_path, model, tokenizer):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(save_zen_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_zen_model_path, CONFIG_NAME)
    # output_dict_file = os.path.join(save_zen_model_path, "dict.bin")
    # output_vocab_file = os.path.join(save_zen_model_path, VOCAB_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    # torch.save({
    #     "labels_dict": processor.labels_dict,
    #     "types_dict": processor.types_dict
    #     "link_metric":
    # }, output_dict_file)
    # with open(output_config_file, "w", encoding='utf-8') as writer:
    #    writer.write(model_to_save.config.to_json_string())
    # tokenizer.save(output_vocab_file)


def store_graph(graph):
    dir = os.path.join(graph.data_dir, "{}.pkl".format("graph"))
    with open(dir, "wb") as gh:
        pickle.dump(graph, gh)


def load_graph(dir="./data/pre_english"):
    graph_path = os.path.join(dir, "{}.pkl".format("graph"))
    with open(graph_path, "rb") as gh:
        g = pickle.load(gh)
        return g


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0