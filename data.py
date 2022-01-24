import copy
import json
import torch
import math

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


def transform_offsets(start, end, offsets_list):
    curr_list = offsets_list[1:-1].copy()
    length = len(offsets_list)
    curr_list.append((math.inf, math.inf))

    start_idx, end_idx = 0, 1

    for i in range(length - 1):
        if start > curr_list[i][0] and start <= curr_list[i+1][0]:
            start_idx = i+1
        if end > curr_list[i][0] and end <= curr_list[i+1][0]:
            end_idx = i+1

    return start_idx, end_idx


def transform_to_list(start, end, seq_length):
    output_list = [0.0 for _ in range(seq_length)]
    for i in range(start, end):
        output_list[i] = 1.0
    return output_list


class ZSLDataset(Dataset):
    def __init__(self, path, tokenizer, train_event_onto, max_len=128):
        self.data = []
        self.role_type_idxs = self.generate_role_idxs(train_event_onto) # {"Attacker": 1}
        self.role_type_num = len(self.role_type_idxs)

        self.event_type_idxs = self.generate_event_idxs(train_event_onto) # {"Conflict:Attack": 1}
        self.rev_event_type_idxs = {v:k for k,v in self.event_type_idxs.items()} # {1: "Conflict:Attack"}

        self.train_ontology = train_event_onto
        self.max_len = max_len

        # load data instance
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        max_pair_num = 1
        overlength_num = 0

        for line in lines:
            data_item = json.loads(line)
            new_data_item = {}

            bert_inputs = tokenizer(data_item["sentence"], return_offsets_mapping=True)
            if len(bert_inputs["input_ids"]) > self.max_len:
                overlength_num += 1
                continue
            new_data_item["original_data"] = data_item
            new_data_item["input_ids"] = bert_inputs["input_ids"]
            new_data_item["attn_mask"] = bert_inputs["attention_mask"]

            seq_len = len(bert_inputs["input_ids"])

            offset_mapping = bert_inputs["offset_mapping"]
            role_pairs = []
            neg_labels = []
            event_type_idxs = []

            for role in data_item["roles"]:
                if role[-2] in self.train_ontology:
                    role_name = role[-1]
                    # if role_name != "unrelated object": 

                    trig_s, trig_e = transform_offsets(role[0][0], role[0][1], offset_mapping)
                    ent_s, ent_e = transform_offsets(role[1][0], role[1][1], offset_mapping)
                    
                    trig_list = transform_to_list(trig_s+1, trig_e+1, seq_len)
                    ent_list = transform_to_list(ent_s+1, ent_e+1, seq_len)
                
                    event_type_idxs.append(self.event_type_idxs[role[-2]])
                    role_idx = self.role_type_idxs[role_name]
                    role_pairs.append([trig_list, ent_list, role_idx])

                    neg_labels_role = []
                    if role_idx != -1:
                        for k in range(self.role_type_num - 1):
                            if k != role_idx:
                                neg_labels_role.append(k)
                    else:
                        for k in range(self.role_type_num - 2):
                            neg_labels_role.append(k)

                    neg_labels.append(neg_labels_role.copy())
                    if len(neg_labels_role) == 17:
                        print(len(neg_labels_role))
            
            if len(role_pairs) > 0:
                new_data_item["pairs"] = deepcopy(role_pairs)
                new_data_item["neg_labels"] = deepcopy(neg_labels)
                new_data_item["trigger_idxs"] = event_type_idxs.copy()

                self.data.append(new_data_item)
                if len(role_pairs) > max_pair_num:
                    max_pair_num = len(role_pairs)

        # print(self.role_type_idxs)
        # print(max_pair_num)
        # print(overlength_num)
        # print(len(self.data))              

    def generate_role_idxs(self, train_event_onto):
        role_type_idxs = {"unrelated object": -1}
        role_type_num = 0
        # role_type_idxs = {}
        # role_type_num = 0
        for event_type in train_event_onto:
            role_types = train_event_onto[event_type]
            for role in role_types:
                if role not in role_type_idxs:
                    role_type_idxs[role] = role_type_num
                    role_type_num += 1
        return role_type_idxs
    
    def generate_event_idxs(self, train_event_onto):
        event_type_idxs = {}
        event_type_num = 0
        for event_type in train_event_onto:
            if event_type not in event_type_idxs:
                event_type_idxs.update({event_type: event_type_num})
                event_type_num += 1
        return event_type_idxs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        # for item in batch:
        #     print(item["original_data"])
        # print('\n')
        
        # batch: a list of data items, "input_ids", "attn_mask", "pairs"
        # print(batch)
        batch_input_ids, batch_attn_mask = [], []
        batch_label_idxs, batch_pair_mask = [], []
        batch_trigger_span, batch_entity_span = [], []
        batch_neg_labels = []
        batch_trigger_idxs = []
        batch_gold_labels = []

        batch_train_pair_mask = []

        max_pairs_num = max([len(item["pairs"]) for item in batch])
        max_seq_len = max([len(item["input_ids"]) for item in batch])

        for inst in batch:
            batch_input_ids.append(inst["input_ids"] + [0 for _ in range(max_seq_len-len(inst["input_ids"]))])
            batch_attn_mask.append(inst["attn_mask"] + [0 for _ in range(max_seq_len-len(inst["attn_mask"]))])

            pairs = inst["pairs"]
            neg_labels = inst["neg_labels"]
            trigger_idxs = inst["trigger_idxs"]

            batch_triggers, batch_entities = [], []
            batch_labels, batch_mask, train_pair_mask = [], [], []
            neg_label_idxs = []
            trigger_idxs_i = []
            gold_labels_i = []

            for i in range(max_pairs_num):
                if i < len(pairs):
                    batch_triggers.append(pairs[i][0]+[0.0 for _ in range(max_seq_len-len(pairs[i][0]))])
                    batch_entities.append(pairs[i][1]+[0.0 for _ in range(max_seq_len-len(pairs[i][1]))])
                    if pairs[i][-1] == -1:
                        batch_labels.append(0)
                        train_pair_mask.append(0)
                    else:
                        batch_labels.append(pairs[i][-1])
                        train_pair_mask.append(1)
                    trigger_idxs_i.append(trigger_idxs[i])
                    neg_label_idxs.append(neg_labels[i])
                    gold_labels_i.append(pairs[i][-1])
                    batch_mask.append(1)
                    

                else:
                    trigger_holder = [0.0 for _ in range(max_seq_len)]
                    trigger_holder[0] = 1.0
                    entity_holder = [0.0 for _ in range(max_seq_len)]
                    entity_holder[0] = 1.0
                    batch_triggers.append(trigger_holder.copy())
                    batch_entities.append(entity_holder.copy())
                    batch_labels.append(0)
                    trigger_idxs_i.append(0)
                    neg_label_idxs.append([0 for _ in range(self.role_type_num - 2)])
                    batch_mask.append(0)
                    train_pair_mask.append(0)
            
            batch_trigger_span.append(deepcopy(batch_triggers))
            batch_entity_span.append(deepcopy(batch_entities))

            batch_label_idxs.append(batch_labels.copy())
            batch_pair_mask.append(batch_mask.copy())

            batch_neg_labels.append(deepcopy(neg_label_idxs))
            batch_trigger_idxs.append(trigger_idxs_i.copy())

            batch_gold_labels.append(gold_labels_i.copy())

            batch_train_pair_mask.append(train_pair_mask.copy())


        batch_input_ids = torch.LongTensor(batch_input_ids)           
        batch_attn_mask = torch.LongTensor(batch_attn_mask)           

        batch_trigger_span = torch.FloatTensor(batch_trigger_span)
        batch_entity_span = torch.FloatTensor(batch_entity_span)

        batch_label_idxs = torch.LongTensor(batch_label_idxs)
        # print([len(for labels in batch_neg_labels])
        batch_neg_labels = torch.LongTensor(batch_neg_labels)
        batch_trigger_idxs = torch.LongTensor(batch_trigger_idxs)
        batch_pair_mask = torch.FloatTensor(batch_pair_mask)
        batch_train_pair_mask = torch.FloatTensor(batch_train_pair_mask)

        new_batch = {
            "input_ids": batch_input_ids,
            "attn_mask": batch_attn_mask,
            "trigger_spans": batch_trigger_span,
            "entity_spans": batch_entity_span,
            "trigger_idxs": batch_trigger_idxs,
            "label_idxs": batch_label_idxs,
            "neg_label_idxs": batch_neg_labels,
            "pair_mask": batch_pair_mask,
            "train_pair_mask": batch_train_pair_mask,
            "gold_labels": batch_gold_labels,
            "max_pair_num": max_pairs_num
        }

        return new_batch


if __name__ == "__main__":
    # s = [(0, 0), (0, 1), (2, 6), (7, 9), (10, 14), (15, 19), (19, 20), (20, 21), (0, 0)]
    # start, end = transform_offsets(8, 13, s)
    # print(start, end)

    # event_types = ['Business:Declare-Bankruptcy', 'Business:End-Org', 'Business:Merge-Org', 'Business:Start-Org', 'Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write', 'Justice:Acquit', 'Justice:Appeal', 'Justice:Arrest-Jail', 'Justice:Charge-Indict', 'Justice:Convict', 'Justice:Execute', 'Justice:Extradite', 'Justice:Fine', 'Justice:Pardon', 'Justice:Release-Parole', 'Justice:Sentence', 'Justice:Sue', 'Justice:Trial-Hearing', 'Life:Be-Born', 'Life:Die', 'Life:Divorce', 'Life:Injure', 'Life:Marry', 'Movement:Transport', 'Personnel:Elect', 'Personnel:End-Position', 'Personnel:Nominate', 'Personnel:Start-Position', 'Transaction:Transfer-Money', 'Transaction:Transfer-Ownership']

    # role_types = ['Adjudicator', 'Agent', 'Artifact', 'Attacker', 'Beneficiary', 'Buyer', 'Defendant', 'Destination', 'Entity', 'Giver', 'Instrument', 'Org', 'Origin', 'Person', 'Place', 'Plaintiff', 'Prosecutor', 'Recipient', 'Seller', 'Target', 'Vehicle', 'Victim']

    # test_event_types = ['Justice:Pardon', 'Justice:Extradite', 'Justice:Acquit', 'Personnel:Nominate', 'Business:Merge-Org', 'Justice:Execute', 'Justice:Fine', 'Life:Divorce', 'Business:Declare-Bankruptcy', 'Business:End-Org', 'Justice:Release-Parole', 'Justice:Appeal', 'Business:Start-Org', 'Life:Be-Born', 'Justice:Sue', 'Justice:Convict', 'Life:Marry', 'Conflict:Demonstrate', 'Justice:Arrest-Jail', 'Justice:Sentence', 'Justice:Charge-Indict', 'Justice:Trial-Hearing', 'Personnel:Start-Position']

    # train_event_types = ['Conflict:Attack', 'Contact:Meet', 'Contact:Phone-Write', 'Life:Die', 'Life:Injure', 'Movement:Transport', 'Personnel:Elect', 'Personnel:End-Position', 'Transaction:Transfer-Money', 'Transaction:Transfer-Ownership']

    with open("./data/train_ontology.json", 'r', encoding="utf-8") as f:
        train_o = json.loads(f.read())
    
    with open("./data/test_ontology.json", 'r', encoding="utf-8") as f:
        test_o = json.loads(f.read())

    # print(train_o)
    # print('\n')

    data_dir = "./data/train.json"
    t = BertTokenizerFast.from_pretrained("bert-large-uncased")
    d = ZSLDataset(data_dir, t, train_o)
    print(d.role_type_idxs)
    i = 0
    for batch in DataLoader(d, batch_size=3, shuffle=True, collate_fn=d.collate_fn):
        # print(batch["neg_label_idxs"])
        # print(batch["label_idxs"])
        # print(batch)
        i += 1
        if i > 0:
            break

    

    
