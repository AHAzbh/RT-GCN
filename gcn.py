import json
import os
import time

import torch
import utils
import copy
import scipy.sparse as sp
from loss import FocalLoss
from layer import layer
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from self_attention import SelfAttention



class GCN(nn.Module):
    def __init__(self, data_dir, graph, gcn_features, normal_features, link_mode, plm, tokenizer, batch_size,
                 hidden_dropout_prob, num_attention_heads, device, config=None):
        super(GCN, self).__init__()
        self._init_graph(graph)

        self.data_dir = data_dir
        self.plm = plm
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.gcn_feature, self.normal_feature = [], []
        self.fixed_embeddings = self._get_fixed_embeddings()

        self.self_link = torch.eye(self.node_count)
        # self.link_matrix = self._build_matrix_v2()  # self._build_matrix_v1()
        if link_mode == 1:
            self.link_matrix = torch.tensor(self.link_matrix_1).repeat(self.batch_size, 1, 1)
        elif link_mode == 2:
            self.link_matrix = torch.tensor(self.link_matrix_2).repeat(self.batch_size, 1, 1)
        else:
            self.link_matrix = torch.tensor(self.link_matrix_0).repeat(self.batch_size, 1, 1)

        self.gcn_feature.extend(gcn_features)
        self.normal_feature.extend(normal_features)
        self.gcn_layer = self._build_gcn_layer(hidden_dropout_prob, num_attention_heads)
        self.normal_layer = self._build_normal_layer()
        self.attention_layer = SelfAttention(self.gcn_feature[-1] * self.node_count,
                                             self.gcn_feature[-1] * self.node_count,
                                             num_attention_heads, hidden_dropout_prob)

        self.dropout = nn.Dropout(hidden_dropout_prob)
        # self.classifier_2 = nn.Linear(gcn_features[-1] * self.node_count + normal_features[-1], self.label_count)
        self.classifier_2 = nn.Linear(self.gcn_feature[-1] * self.node_count + self.normal_feature[-1], self.label_count)

        # self.CE_weight = torch.ones(self.label_count) * 10
        # self.CE_weight[self.label_index["Other"]] = 1.
        # self.loss_fct = CrossEntropyLoss(weight=self.CE_weight)
        # self.focal_loss = FocalLoss(class_num=self.label_count, alpha=self.CE_weight)

    def _init_graph(self, graph):
        self.direct = graph.direct
        self.node_count = graph.node_count
        self.relation_count = graph.relation_count
        self.label_count = graph.label_count
        # self.first_layer, self.second_layer = graph.first_layer, graph.second_layer
        # self.first_layer_count, self.second_layer_count = graph.first_layer_count, graph.second_layer_count
        # self.node_name_list = []
        self.node_index = graph.node_index
        self.label_index = graph.label_index
        # self.links = graph.links
        self.fixed_node = graph.fixed_node
        self.link_matrix_0 = graph.link_matrix_0
        self.link_matrix_1, self.link_matrix_2 = graph.link_matrix_1, graph.link_matrix_2

    def _label_embedding(self, sentence, tokens, token_mask, node_name):
        if len(token_mask) != len(node_name):
            raise Exception("label length error")
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        label_mask = utils.get_label_mask(tokens, token_mask, self.tokenizer)
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        last_hidden_state = self.plm(**inputs).last_hidden_state
        label_embeddings = []
        for ind in range(len(label_mask)):
            label = dict()
            embedding = last_hidden_state[0][label_mask[ind][0] + 1:label_mask[ind][1] + 1]
            embedding = embedding.sum(dim=0) / (label_mask[ind][1] - label_mask[ind][0])
            # print("label embedding device {}".format(embedding.device))
            label["embedding"] = embedding
            label["node_name"] = node_name[ind]
            label_embeddings.append(label)

        return label_embeddings

    def _get_fixed_embeddings(self):
        embeddings = []
        for s in self.fixed_node:
            embedding = self._label_embedding(s["sentence"], s["tokens"], s["masks"], s["node_name"])
            embeddings.extend(embedding)

        feature_0 = embeddings[0]["embedding"].shape[0]
        self.gcn_feature.append(feature_0)
        self.normal_feature.append(feature_0 * 3)
        fixed_embedding = torch.zeros(self.node_count, feature_0)
        # fixed_embeddings = torch.zeros(self.batch_size, self.node_count, feature_0)
        for e in embeddings:
            node_name = e["node_name"]
            if node_name in self.node_index:
                fixed_embedding[self.node_index[node_name]] = e["embedding"]
            elif "SUB-" + node_name in self.node_index:
                fixed_embedding[self.node_index["SUB-" + node_name]] = e["embedding"]
                # elif "OBJ-" + node_name in self.node_index:
                fixed_embedding[self.node_index["OBJ-" + node_name]] = e["embedding"]
        # print("solid embeddings device {}".format(fixed_embedding.device))

        fixed_embeddings = fixed_embedding.repeat(self.batch_size, 1, 1)

        return fixed_embeddings

    def _build_gcn_layer(self, hidden_dropout_prob, num_attention_heads):
        feature_count = len(self.gcn_feature)
        layers = []
        for i in range(feature_count - 1):
            single_layer = nn.ModuleList(
                [copy.deepcopy(layer(self.gcn_feature[i], self.gcn_feature[i + 1])),
                 copy.deepcopy(layer(self.gcn_feature[i], self.gcn_feature[i + 1])),
                 copy.deepcopy(layer(self.gcn_feature[i], self.gcn_feature[i + 1]))]
            )
            if self.gcn_feature[i + 1] <= 10:
                single_layer.append(copy.deepcopy(SelfAttention(self.gcn_feature[i + 1] * self.node_count,
                                                                self.gcn_feature[i + 1] * self.node_count,
                                                                num_attention_heads, hidden_dropout_prob)))
            layers.append(single_layer)
        return nn.ModuleList(layers)

    def _build_normal_layer(self):
        feature_count = len(self.normal_feature)
        layers = []
        for i in range(feature_count - 1):
            single_layer = copy.deepcopy(layer(self.normal_feature[i], self.normal_feature[i + 1]))
            layers.append(single_layer)
        return nn.ModuleList(layers)

    def _build_input(self, sentences, label_masks, sub_types, obj_types, batch_size):
        token_ids = self.tokenizer(sentences, return_tensors="pt", padding=True).to(self.device)
        last_hidden_state = self.plm(**token_ids).last_hidden_state
        sub_embeddings, obj_embeddings = torch.zeros(batch_size, last_hidden_state.size(-1)), \
                                         torch.zeros(batch_size, last_hidden_state.size(-1))
        sub_start, sub_end = label_masks[0][0], label_masks[0][1]
        obj_start, obj_end = label_masks[1][0], label_masks[1][1]
        for i in range(batch_size):
            sub_embedding = last_hidden_state[i][sub_start[i] + 1:sub_end[i] + 1]
            sub_embedding = sub_embedding.sum(dim=0) / (sub_end[i] - sub_start[i])
            obj_embedding = last_hidden_state[i][obj_start[i] + 1:obj_end[i] + 1]
            obj_embedding = obj_embedding.sum(dim=0) / (obj_end[i] - obj_start[i])
            sub_embeddings[i] = sub_embedding
            obj_embeddings[i] = obj_embedding

        cls = last_hidden_state[:, 0]

        inputs = self.fixed_embeddings.data
        link_matrix = self.link_matrix.data

        si, oi = self.node_index["SUB-INPUT"], self.node_index["OBJ-INPUT"]
        inputs[:, si], inputs[:, oi] = sub_embeddings, obj_embeddings
        sub_link = torch.zeros(len(sub_types), self.node_count)
        obj_link = torch.zeros(len(obj_types), self.node_count)

        sub_link = sub_link.scatter(1, sub_types.view(-1, 1).data, 1.)
        obj_link = obj_link.scatter(1, obj_types.view(-1, 1).data, 1.)

        link_matrix[:, :, si] = link_matrix[:, si, :] = sub_link
        link_matrix[:, :, oi] = link_matrix[:, oi, :] = obj_link

        return inputs[0:batch_size], link_matrix[0:batch_size], cls, sub_embeddings, obj_embeddings

    def forward(self, sentences, label_masks, sub_types, obj_types):
        batch_size = len(sentences)
        input_embeddings, link_matrix, cls, sub_embeddings, obj_embeddings = self._build_input(sentences, label_masks, sub_types, obj_types, batch_size)
        # print("input embeddings device {}".format(input_embeddings.device))
        # print("link matrix device {}".format(link_matrix.device))
        # print("input shape {}".format(input_embeddings.shape))
        input_embeddings = input_embeddings.to(self.device)
        link_matrix = link_matrix.to(self.device)
        sub_embeddings = sub_embeddings.to(self.device)
        obj_embeddings = obj_embeddings.to(self.device)
        output = self.dropout(input_embeddings)
        # print("link_matrix", link_matrix)
        # print("output device {}".format(output.device))
        for i, layer_module in enumerate(self.gcn_layer):
            # self_output = torch.matmul(self.self_link, output)
            # print("output device {}".format(output.device))
            self_output = layer_module[0](output).reshape(batch_size, -1)

            link_output_1 = layer_module[1](torch.matmul(link_matrix, output)).reshape(batch_size, -1)
            # print("link_output1", link_output_1)
            link_output_2 = layer_module[2](torch.matmul(link_matrix, torch.matmul(link_matrix, output))).reshape(batch_size, -1)
            # print("link_output2", link_output_2)
            attention_input = torch.cat((self_output, link_output_1, link_output_2), dim=-1).reshape(batch_size, 3, -1)
            if len(layer_module) == 4:
                attention = layer_module[3](attention_input)
            else:
                attention = attention_input
            output = torch.mean(attention, dim=1).reshape(batch_size, self.node_count, -1)
            # output = torch.add(torch.add(self_output, link_output_1), link_output_2)
            # print("output", output)
            # output = self.dropout(output)
            # print("output device {}".format(output.device))
        # print("gcn output shape is {}".format(output.shape))

        gcn_embedding = output.reshape(batch_size, -1)
        # print("gcn embedding shape is {}".format(gcn_embedding.shape))

        normal_input = torch.cat((cls, sub_embeddings, obj_embeddings), dim=-1)
        output = self.dropout(normal_input)
        for i, layer_module in enumerate(self.normal_layer):
            output = self.dropout(layer_module(output))

        # print("cls output shape is {}".format(output.shape))

        # output = torch.cat((cls, sub_embeddings, obj_embeddings), dim=-1)

        normal_embedding = output.reshape(batch_size, -1)
        # print("cls embedding shape is {}".format(normal_embedding.shape))
        all_embedding = torch.cat((normal_embedding, gcn_embedding), dim=-1)
        # print("all embedding shape is {}".format(all_embedding.shape))

        # print(output.shape)
        # logits = torch.softmax(self.classifier(output).split(self.relation_count, dim=1)[0].squeeze(), dim=-1)
        # logits = torch.softmax(self.classifier_2(output.squeeze(-1)), dim=-1)
        # print("all shape {}".format(all_embedding.shape))
        # print(self.classifier_2)
        logits = self.classifier_2(all_embedding)

        return logits

