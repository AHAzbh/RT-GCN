import copy
import json
from data_util import PromptDataProcessor
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt.plms import load_plm
from openprompt.prompts import SoftVerbalizer
import torch
from openprompt.data_utils.utils import InputExample
from transformers import AdamW, get_linear_schedule_with_warmup


def prompt(examples_list):
    processor = PromptDataProcessor("./data/ace")
    classes = processor.get_labels()
    print(processor.get_num_labels())
    print(classes)
    label_words = dict()
    for item in classes:
        label_words[item] = [item]
    train_examples = processor.get_train_examples()
    test_examples = processor.get_test_examples()
    # config, args = cfg.get_config()
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"} {"placeholder":"text_b"} is {"mask"}',
        tokenizer=tokenizer,
    )

    # wrapped_example = promptTemplate.wrap_one_example(train_examples[0])

    myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=len(classes), label_words=classes)

    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words=label_words,
        tokenizer=tokenizer,
    )

    train_dataloader = PromptDataLoader(
        dataset=train_examples, template=promptTemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")

    test_dataloader = PromptDataLoader(
        dataset=test_examples, template=promptTemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=myverbalizer,
        freeze_plm=False,
    )
    use_cuda = True
    if use_cuda:
        promptModel = promptModel.cuda()


    loss_func = torch.nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in promptModel.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer_grouped_parameters2 = [
        {'params': promptModel.verbalizer.group_parameters_1, "lr": 3e-5},
        {'params': promptModel.verbalizer.group_parameters_2, "lr": 3e-4},
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2)


    for epoch in range(5):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = promptModel(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            print(tot_loss / (step + 1))

    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = promptModel(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    print("test:", acc)  # roughly ~0.85

    pred_result = []
    for examples in examples_list:
        pred_ids, pred_labels = [], []
        pred_dataloader = PromptDataLoader(
            dataset=examples, template=promptTemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
            batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head")
        for step, inputs in enumerate(pred_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = promptModel(inputs)
            pred_ids.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        for id in pred_ids:
            pred_labels.append(processor.id2label[id])
        pred_result.append(pred_labels)
    return pred_result

    # runner = ClassificationRunner(
    #     model=promptModel,
    #     config=config,
    #     train_dataloader=train_dataloader,
    #     test_dataloader=test_dataloader,
    # )
    #
    # runner.run(ckpt="./model")


def pre_file(filepath):
    train = list()
    with open(filepath) as tr:
        while 1:
            line = tr.readline()
            if line is None or line == "":
                break
            pre_tokens = line.split("\t")
            print(pre_tokens)
            e1, e2, relation = pre_tokens[0], pre_tokens[1], pre_tokens[2]

            tokens = list()
            item = dict()
            item["relation"] = dict()
            item["tokens"] = list()
            e1_s, e1_e, e2_s, e2_e = 0, 0, 0, 0
            pre_tokens = pre_tokens[3].split(" ")
            for i, token in enumerate(pre_tokens):
                if token == "<e1>":
                    e1_s = i
                elif token == "</e1>":
                    e1_e = i - 1
                elif token == "<e2>":
                    e2_s = i - 2
                elif token == "</e2>":
                    e2_e = i - 3
                else:
                    if i == len(pre_tokens) - 1 and token == ".\n":
                        token = "."
                    tokens.append(token)

            # e1 = " ".join(tokens[e1_s:e1_e])
            # e2 = " ".join(tokens[e2_s:e2_e])
            # re_ind = e1_e + e2_e - e1_s - e2_s
            # relation = tokens[re_ind]
            sub, obj = e1, e2
            sub_s = e1_s
            sub_e = e1_e
            obj_s = e2_s
            obj_e = e2_e
            print(e1, e2, relation)

            if relation == "other":
                item["relation"]["relation_type"] = "Other"
            else:
                re_split = relation.split("(")
                item["relation"]["relation_type"] = re_split[0]
                e1_p = re_split[1].index("e1")
                e2_p = re_split[1].index("e2")
                if e1_p > e2_p:
                    sub, obj = e2, e1
                    sub_s = e2_s
                    sub_e = e2_e
                    obj_s = e1_s
                    obj_e = e1_e
            item["text"] = " ".join(tokens)
            item["tokens"] = tokens
            item["relation"]["Arg-1"] = dict()
            item["relation"]["Arg-2"] = dict()
            item["relation"]["Arg-1"]["text"] = sub
            item["relation"]["Arg-2"]["text"] = obj
            item["relation"]["Arg-1"]["entity"] = dict()
            item["relation"]["Arg-2"]["entity"] = dict()
            item["relation"]["Arg-1"]["entity"]["text"] = sub
            item["relation"]["Arg-2"]["entity"]["text"] = obj
            item["relation"]["Arg-1"]["entity"]["start"] = sub_s
            item["relation"]["Arg-1"]["entity"]["end"] = sub_e
            item["relation"]["Arg-2"]["entity"]["start"] = obj_s
            item["relation"]["Arg-2"]["entity"]["end"] = obj_e
            print(item)
            train.append(item)
        return train


def prompt_input(data_list):
    result = []
    for data in data_list:
        tokens = data["tokens"]
        entities = [data["relation"]["Arg-1"]["entity"], data["relation"]["Arg-2"]["entity"]]
        for entity in entities:
            new_token = copy.deepcopy(tokens)
            start, end = entity["start"], entity["end"]
            new_token.insert(end, "[/e]")
            new_token.insert(start, "[e]")
            text = " ".join(new_token)
            mention = entity["text"]
            item = dict()
            item["text_a"] = text
            item["text_b"] = mention
            result.append(item)

    examples = list()
    for i, data in enumerate(result):
        example = InputExample(
            guid=i,
            text_a=data["text_a"],
            text_b=data["text_b"],
        )
        examples.append(example)
    return examples


def use_prompt():
    train = pre_file("./data/semeval/train.txt")
    test = pre_file("./data/semeval/test.txt")
    train_input = prompt_input(train)
    test_input = prompt_input(test)
    result = prompt([train_input, test_input])
    train_pred, test_pred = result[0], result[1]
    for i, pred in enumerate(train_pred):
        ind = i // 2
        arg = i % 2
        res = pred.split(":")
        if arg == 0:
            train[ind]["relation"]["Arg-1"]["entity"]["entity_type"] = res[0]
            train[ind]["relation"]["Arg-1"]["entity"]["entity_subtype"] = res[1]
        else:
            train[ind]["relation"]["Arg-2"]["entity"]["entity_type"] = res[0]
            train[ind]["relation"]["Arg-2"]["entity"]["entity_subtype"] = res[1]

    for i, pred in enumerate(test_pred):
        ind = i // 2
        arg = i % 2
        res = pred.split(":")
        if arg == 0:
            test[ind]["relation"]["Arg-1"]["entity"]["entity_type"] = res[0]
            test[ind]["relation"]["Arg-1"]["entity"]["entity_subtype"] = res[1]
        else:
            test[ind]["relation"]["Arg-2"]["entity"]["entity_type"] = res[0]
            test[ind]["relation"]["Arg-2"]["entity"]["entity_subtype"] = res[1]

    with open("./data/semeval/train.json", "w+") as tr:
        with open("./data/semeval/test.json", "w+") as ts:
            tr.write(json.dumps(train))
            ts.write(json.dumps(test))


if __name__ == '__main__':
    use_prompt()

