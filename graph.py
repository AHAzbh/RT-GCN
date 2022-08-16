import math
import os
import json


class NodeGraph:
    def __init__(self, data_dir, direct=False):
        self.direct = direct
        self.data_dir = data_dir
        self.node_count, self.label_count = 0, 0
        self.relation_count, self.entity_count = 0, 0
        self.node_name_list = []
        self.node_index = dict()
        self.label_index = dict()
        self.fixed_node = []
        self._get_graph()

    def _get_relations(self):
        relations_path = os.path.join(self.data_dir, '{}.json'.format("relations"))
        with open(relations_path) as re:
            self.relations = json.load(re)
            for r in self.relations:
                self.label_index[r] = self.label_count
                self.label_count += 1
                for rr in self.relations[r]:
                    re_name = r + ":" + rr
                    self.node_index[re_name] = self.node_count
                    self.node_name_list.append(re_name)
                    self.node_count += 1

                    node = dict()
                    relation_name = r + ":" + rr
                    node["node_name"] = [relation_name]
                    node["sentence"] = rr
                    node["tokens"] = [rr]
                    node["masks"] = [(0, 1)]
                    self.fixed_node.append(node)

            # self.node_index["Other"] = self.node_count
            self.label_index["Other"] = self.label_count
            # self.node_name_list.append("Other")
            # self.node_count += 1
            self.label_count += 1
            self.relation_count = self.node_count

    def _get_entity_type(self, template=" includes "):
        entity_type_path = os.path.join(self.data_dir, '{}.json'.format("entity_type"))
        with open(entity_type_path) as en:
            self.entity_type = json.load(en)
            for fe in self.entity_type:
                node = dict()
                sentence = fe + template
                masks = list()
                type_names = list()
                tokens = [fe]
                template_words = template.split(" ")
                length = len(template_words)
                for t in template_words:
                    if t == "":
                        length -= 1
                    else:
                        tokens.append(t)

                start, end = 1 + length, 1 + length
                for se in self.entity_type[fe]:
                    start = end
                    end += 1
                    masks.append((start, end))
                    en_name = fe + ":" + se
                    type_names.append(en_name)
                    tokens.append(se)
                    sentence += se + ", "
                    tokens.append(",")
                    end += 1

                    sub_name = "SUB-" + en_name
                    obj_name = "OBJ-" + en_name
                    self.node_name_list.extend([sub_name, obj_name])
                    self.node_index[sub_name] = self.node_count
                    self.node_count += 1
                    self.node_index[obj_name] = self.node_count
                    self.node_count += 1

                node["sentence"] = sentence
                node["tokens"] = tokens
                node["masks"] = masks
                node["node_name"] = type_names
                self.fixed_node.append(node)
            self.entity_count = self.node_count - self.relation_count

    def _build_matrix(self, res):
        subs, objs, count = res["subject"], res["object"], res["count"]
        link_matrix = [[0.0 for _ in range(self.node_count)] for _ in range(self.node_count)]
        # sub_tota, obj_tota = 0, 0
        # for sub in subs:
        #     sub_tota += subs[sub]["count"]
        # for obj in objs:
        #     obj_tota += objs[obj]["count"]
        # print(sub_tota, obj_tota)
        re_sub_count = dict()
        for sub in subs:
            for re_name in subs[sub]:
                if re_name == "count":
                    continue
                if re_name in re_sub_count:
                    re_sub_count[re_name] += subs[sub][re_name]
                else:
                    re_sub_count[re_name] = subs[sub][re_name]

        re_obj_count = dict()
        for obj in objs:
            for re_name in objs[obj]:
                if re_name == "count":
                    continue
                if re_name in re_obj_count:
                    re_obj_count[re_name] += objs[obj][re_name]
                else:
                    re_obj_count[re_name] = objs[obj][re_name]

        for sub in subs:
            sub_name = "SUB-" + sub
            for re_name in subs[sub]:
                if re_name == "count":
                    continue
                i, j = self.node_index[sub_name], self.node_index[re_name]
                link_matrix[i][j] = link_matrix[j][i] = \
                    float(subs[sub][re_name]) / math.sqrt(subs[sub]["count"] * re_sub_count[re_name])
                    # float(subs[sub][re_name]) / subs[sub]["count"] / self.relation_count  # len(subs)

        for obj in objs:
            obj_name = "OBJ-" + obj
            for re_name in objs[obj]:
                if re_name == "count":
                    continue
                i, j = self.node_index[obj_name], self.node_index[re_name]
                link_matrix[i][j] = link_matrix[j][i] = \
                    float(objs[obj][re_name]) / math.sqrt(objs[obj]["count"] * re_obj_count[re_name])
                    # float(objs[obj][re_name]) / objs[obj]["count"] / self.relation_count  # len(objs)

        return link_matrix

    def _get_matrix(self):
        relations_path = os.path.join(self.data_dir, '{}.json'.format("relations_2"))
        with open(relations_path) as re:
            relations = json.load(re)
            re_v1, re_v2 = relations["v1"], relations["v2"]
            self.link_matrix_1 = self._build_matrix(re_v1)
            self.link_matrix_2 = self._build_matrix(re_v2)

    def _get_graph(self):
        self._get_relations()
        self._get_entity_type()
        self.node_name_list.extend(["SUB-INPUT", "OBJ-INPUT"])
        self.node_index["SUB-INPUT"] = self.node_count
        self.node_count += 1
        self.node_index["OBJ-INPUT"] = self.node_count
        self.node_count += 1
        self._get_matrix()
        self.link_matrix_0 = self._build_matrix_v1()

    def _build_re_v1(self):
        self.links = []
        self.first_layer_count, self.second_layer_count = 0, 0
        self.first_layer, self.second_layer = dict(), dict()
        for r in self.relations:
            for rr in self.relations[r]:
                re_name = r + ":" + rr

                for s_o in self.relations[r][rr]:
                    sub = "SUB-" + s_o["subject"]
                    obj = "OBJ-" + s_o["object"]
                    self.links.append((self.node_index[sub], self.node_index[re_name]))
                    self.first_layer_count += 1
                    link_name = str(self.node_index[sub]) + ":" + str(self.node_index[re_name])
                    if link_name in self.first_layer:
                        self.first_layer[link_name] += 1
                    else:
                        self.first_layer[link_name] = 1

                    self.links.append((self.node_index[obj], self.node_index[re_name]))
                    link_name = str(self.node_index[obj]) + ":" + str(self.node_index[re_name])
                    if link_name in self.second_layer:
                        self.second_layer[link_name] += 1
                    else:
                        self.second_layer[link_name] = 1
                    self.second_layer_count += 1

    def _build_matrix_v1(self):
        self._build_re_v1()
        link_matrix = [[0.0 for _ in range(self.node_count)] for _ in range(self.node_count)]

        for i in self.links:
            link_matrix[i[0]][i[1]] += 1
            link_matrix[i[1]][i[0]] += 1

        for i in range(self.relation_count):
            for j in range(self.relation_count, self.node_count):
                sub = j - self.relation_count
                if sub % 2:
                    link_matrix[i][j] = float(link_matrix[i][j]) / self.second_layer_count
                else:
                    link_matrix[i][j] = float(link_matrix[i][j]) / self.first_layer_count

        for i in range(self.relation_count, self.node_count):
            sub = i - self.relation_count
            for j in range(self.relation_count):
                if sub % 2:
                    link_matrix[i][j] = float(link_matrix[i][j]) / self.second_layer_count
                else:
                    link_matrix[i][j] = float(link_matrix[i][j]) / self.first_layer_count

        return link_matrix
