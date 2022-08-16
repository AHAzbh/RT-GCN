import os
import json

from torch.utils.data import Dataset
import utils


class REDataset(Dataset):
    def __init__(self, features, node_index, label_index, tokenizer, device):
        self.data = features
        # self.plm = plm
        self.node_index = node_index
        self.label_index = label_index
        self.tokenizer = tokenizer
        self.device = device
        self.embeddings = []

    def __getitem__(self, index):
        token_mask = [self.data[index]["e1_pos"], self.data[index]["e2_pos"]]
        label_mask = utils.get_label_mask(self.data[index]["tokens"], token_mask, self.tokenizer)

        sub_type = self.node_index[self.data[index]["e1_type"]]
        obj_type = self.node_index[self.data[index]["e2_type"]]
        label = self.label_index[self.data[index]["relation_type"]]

        return self.data[index]["text"], label_mask, sub_type, obj_type, label

    def __len__(self):
        return len(self.data)


class REProcessor:
    def __init__(self, data_dir, node_index, label_index, tokenizer, device):
        self.data_dir = data_dir
        self.node_index = node_index
        self.label_index = label_index
        self.tokenizer = tokenizer
        self.device = device

    def get_train_examples(self):
        features = create_features(os.path.join(self.data_dir, '{}.json'.format("train")))
        return REDataset(features, self.node_index, self.label_index, self.tokenizer, self.device)

    def get_dev_examples(self):
        features = create_features(os.path.join(self.data_dir, '{}.json'.format("dev")))
        return REDataset(features, self.node_index, self.label_index, self.tokenizer, self.device)

    def get_test_examples(self):
        features = create_features(os.path.join(self.data_dir, '{}.json'.format("test")))
        return REDataset(features, self.node_index, self.label_index, self.tokenizer, self.device)


def create_features(data_dir):
    with open(data_dir) as data:
        features = []
        examples = json.load(data)
        for e in examples:
            feature = dict()
            # feature["text"] = e["text"]
            tokens = e["tokens"]
            e1_pos = (e["relation"]["Arg-1"]["entity"]["start"], e["relation"]["Arg-1"]["entity"]["end"])
            e2_pos = (e["relation"]["Arg-2"]["entity"]["start"], e["relation"]["Arg-2"]["entity"]["end"])
            if e1_pos[0] < e2_pos[0]:
                tokens.insert(e2_pos[1], "[/e2]")
                tokens.insert(e2_pos[0], "[e2]")
                tokens.insert(e1_pos[1], "[/e1]")
                tokens.insert(e1_pos[0], "[e1]")
                e1_pos = (e1_pos[0], e1_pos[1] + 1)
                e2_pos = (e2_pos[0] + 2, e2_pos[1] + 3)
            else:
                tokens.insert(e1_pos[1], "[/e1]")
                tokens.insert(e1_pos[0], "[e1]")
                tokens.insert(e2_pos[1], "[/e2]")
                tokens.insert(e2_pos[0], "[e2]")
                e2_pos = (e2_pos[0], e2_pos[1] + 1)
                e1_pos = (e1_pos[0] + 2, e1_pos[1] + 3)

            feature["tokens"] = tokens
            feature["text"] = " ".join(tokens)
            feature["e1"] = e["relation"]["Arg-1"]["entity"]["text"]
            feature["e1_pos"] = e1_pos
            e1_name = e["relation"]["Arg-1"]["entity"]["entity_type"] + ":" \
                      + e["relation"]["Arg-1"]["entity"]["entity_subtype"]
            feature["e1_type"] = "SUB-" + e1_name
            # feature["e1_type"] = self.node_index[e1_name]
            feature["e2"] = e["relation"]["Arg-2"]["entity"]["text"]
            feature["e2_pos"] = e2_pos
            e2_name = e["relation"]["Arg-2"]["entity"]["entity_type"] + ":" \
                      + e["relation"]["Arg-2"]["entity"]["entity_subtype"]
            feature["e2_type"] = "OBJ-" + e2_name
            # feature["e2_type"] = self.node_index[e2_name]
            if e["relation"]["relation_type"] == "Other":
                feature["relation_type"] = e["relation"]["relation_type"]
            else:
                feature["relation_type"] = e["relation"]["relation_type"]  # + ":" + e["relation"]["relation_subtype"]
            # feature["relation_label"] = self.node_index[feature["relation_type"]]
            features.append(feature)
        return features
