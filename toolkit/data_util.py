import os
import json
from typing import *
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.utils import InputExample


class PromptDataProcessor(DataProcessor):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity_type = self.get_entity_type()
        super().__init__(labels=self.entity_type)

    def get_entity_type(self):
        entity_type_path = os.path.join(self.data_dir, '{}.json'.format("entity_type"))
        with open(entity_type_path) as en:
            entity_type = json.load(en)
            type_names = list()
            for fe in entity_type:
                for se in entity_type[fe]:
                    en_name = fe + ":" + se
                    type_names.append(en_name)
        return type_names

    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None) -> List[InputExample]:
        data_path = os.path.join(self.data_dir, '{}.json'.format(split))
        examples = list()
        with open(data_path) as dp:
            data_list = json.load(dp)
            for i, data in enumerate(data_list):
                example = InputExample(
                    guid=i,
                    text_a=data["text_a"],
                    text_b=data["text_b"],
                    label=self.get_label_id(data["class"]),
                )

                examples.append(example)
        return examples






