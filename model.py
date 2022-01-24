from predict import get_spacy_entities, get_noun_entities
import torch
import torch.nn as nn
import numpy as np
import nltk

from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import *
from data import *

def get_numberized_onto(onto, event_type_idx, role_type_idx):
    num_onto = {}
    for event_type in onto:
        et_idx = event_type_idx[event_type]
        roles = onto[event_type]
        num_roles = [role_type_idx[role] for role in roles]
        # num_roles.append(0)
        num_onto.update({et_idx: num_roles.copy()})
    return num_onto


class Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self, dimensions, activation='tanh', dropout_prob=0.3, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


class ZeroShotModel(nn.Module):
    def __init__(self, bert_name, train_role_types, test_role_types, train_onto, test_onto, train_event_types, test_event_types, dropout, bert_dim, ft_hidden_dim, role_hidden_dim, output_dim, alpha, device):
        super(ZeroShotModel, self).__init__()

        self.train_onto = train_onto
        self.test_onto = test_onto
        # ontology: {"Conflict:Attack": ["Attacker", "Victim", "Target", "Place"]}

        self.train_idx_to_type = {v:k for k,v in train_event_types.items()}
        self.test_idx_to_type = {v:k for k,v in test_event_types.items()}
        # idx_to_type: {0: "Conflict:Attack"}

        self.train_event_types = train_event_types
        self.test_event_types = test_event_types
        # event_types: {"Conflict:Attack": 0}

        self.train_role_type_idx = train_role_types # {role_name: idx}
        self.train_role_type_num = len(self.train_role_type_idx) - 1

        self.test_role_type_idx = test_role_types # {role_name: idx}
        self.test_role_type_num = len(self.test_role_type_idx) - 1
        self.test_rev_role_type_idx = {v:k for k,v in self.test_role_type_idx.items()}

        self.numberized_test_onto = get_numberized_onto(test_onto, test_event_types, test_role_types)

        self.device = device
        self.dropout = dropout
        self.bert_dim = bert_dim
        self.ft_hidden_dim = ft_hidden_dim
        self.role_hidden_dim = role_hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.bert = BertModel.from_pretrained(bert_name)
        self.role_name_encoder = Linears([bert_dim, role_hidden_dim, output_dim], dropout_prob=dropout)
        self.text_ft_encoder = Linears([2*bert_dim, ft_hidden_dim, output_dim], dropout_prob=dropout)
        self.cosine_sim_2 = nn.CosineSimilarity(dim=2)

        self.cosine_sim_3 = nn.CosineSimilarity(dim=3)

        self.bert.to(device)

    def compute_train_role_reprs(self, tokenizer):
        role_names = sorted(self.train_role_type_idx.items(), key=lambda x:x[1])
        names = []
        for name in role_names:
            names.append(name[0])
        train_role_reprs = get_bert_embeddings(names, self.bert, tokenizer, self.device)
        self.train_role_reprs = train_role_reprs.detach()[1:, :]
    
    def compute_test_role_reprs(self, tokenizer):
        role_names = sorted(self.test_role_type_idx.items(), key=lambda x:x[1])
        names = []
        for name in role_names:
            names.append(name[0])
        test_role_reprs = get_bert_embeddings(names, self.bert, tokenizer, self.device)
        self.test_role_reprs = test_role_reprs.detach()[1:, :]

    def span_encode(self, bert_output, span_input):
        # bert_output: (batch, seq_len, dim)
        # span_input: (batch, num, seq_len)
        # OUTPUT: (batch, num, dim)
        dim = bert_output.shape[2]
        num = span_input.shape[1]
        avg_span_input = span_input / torch.sum(span_input, 2).unsqueeze(2)
        avg_weights = avg_span_input.unsqueeze(3).repeat(1, 1, 1, dim)
        bert_repeated = bert_output.unsqueeze(1).repeat(1, num, 1, 1)
        span_output = torch.sum(bert_repeated * avg_weights, 2)
        return span_output
    
    def forward(self, batch):
        # batch: {"input_ids", "attn_mask", "trigger_spans", "entity_spans", "label_idxs", "neg_label_idxs", "pair_mask"}
        # pairs_num = batch["pair_mask"].sum()
        bert_outputs = self.bert(batch["input_ids"], attention_mask=batch["attn_mask"])[0]
        trigger_reprs = self.span_encode(bert_outputs, batch["trigger_spans"])
        entity_reprs = self.span_encode(bert_outputs, batch["entity_spans"])
        ta_reprs = torch.cat((trigger_reprs, entity_reprs), 2) 

        # minimize the distance between correct pairs
        role_reprs = self.role_name_encoder(self.train_role_reprs) # (role_num, output_dim)
        label_reprs = role_reprs[batch["label_idxs"]] # (bs, num, output_dim)
        # print() 
        output_ta_reprs = self.text_ft_encoder(ta_reprs) # (bs, num, output_dim)
        pos_cos_sim = self.cosine_sim_2(output_ta_reprs, label_reprs) # (bs, num)

        # print(self.train_role_reprs.shape)
        neg_label_reprs = role_reprs[batch["neg_label_idxs"]] # (bs, num, neg_role_num, output_dim)
        repeated_ta_reprs = output_ta_reprs.unsqueeze(2).repeat(1, 1, self.train_role_type_num-1, 1)

        neg_cos_sim = self.cosine_sim_3(repeated_ta_reprs, neg_label_reprs) # (bs, num, neg_role_num)
        pos_cos_sims = pos_cos_sim.unsqueeze(2).repeat(1, 1, self.train_role_type_num-1)

        hinge_matrix = torch.sum(torch.clamp(neg_cos_sim-pos_cos_sims+self.alpha, min=0), 2)

        hinge_loss = (hinge_matrix * batch["train_pair_mask"]).sum()


        # # min_loss = (self.cosine_sim_2(output_ta_reprs, label_reprs) * batch["pair_mask"]).sum() / pairs_num
        # # maximize the distance between incorrect pairs
        # neg_label_reprs = role_reprs[batch["neg_label_idxs"]] # (bs, num, neg_role_num, output_dim)
        # repeated_ta_reprs = output_ta_reprs.unsqueeze(2).repeat(1, 1, self.train_role_type_num-1, 1)
        # # print(neg_label_reprs.shape)
        # # print(repreated_ta_reprs.shape)
        # neg_cosine_sim = self.cosine_sim_3(repeated_ta_reprs, neg_label_reprs)
        # max_loss = (torch.mean(neg_cosine_sim, 2) * batch["pair_mask"]).sum() / pairs_num

        # # regularization: each cosine vector should be far from each other as far as possible.


        # loss = -min_loss + self.alpha * max_loss
        return hinge_loss

    def predict(self, batch):
        # batch: {"input_ids", "attn_mask", "trigger_spans", "entity_spans", "label_idxs", "neg_label_idxs", "trigger_idxs", "pair_mask"}
        output_list = []
        bs, max_num = batch["trigger_spans"].shape[0], batch["trigger_spans"].shape[1]
        trigger_idxs = batch["trigger_idxs"] # (batch_num, max_num)

        with torch.no_grad():
            bert_outputs = self.bert(batch["input_ids"], attention_mask=batch["attn_mask"])[0]
            trigger_reprs = self.span_encode(bert_outputs, batch["trigger_spans"])
            entity_reprs = self.span_encode(bert_outputs, batch["entity_spans"])
            ta_reprs = torch.cat((trigger_reprs, entity_reprs), 2) 
            output_ta_reprs = self.text_ft_encoder(ta_reprs) # (batch_size, pair_num, output_dim)
            role_reprs = self.role_name_encoder(self.test_role_reprs) # (role_num, output_dim)
            sum_mask = torch.sum(batch["pair_mask"], 1).long().tolist()

            role_num = role_reprs.shape[0]

            for i in range(bs):
                output_i = []
                pair_num_i = sum_mask[i]

                ta_reprs_i = output_ta_reprs[i][0:pair_num_i]
                repeated_ta_reprs_i = ta_reprs_i.unsqueeze(1).repeat(1, role_num, 1)
                repeated_role_reprs = role_reprs.unsqueeze(0).repeat(pair_num_i, 1, 1)

                cos_sim = self.cosine_sim_2(repeated_ta_reprs_i, repeated_role_reprs) # (pair_num_i, role_num)
                # print(cos_sim[0])
                event_type_idx_i = trigger_idxs[i].tolist()
                # print(cos_sim)
                for j in range(pair_num_i):
                    cos_sim_j = cos_sim[j]
                    event_type = event_type_idx_i[j]
                    role_idxs = self.numberized_test_onto[event_type]
                    role_scores = [cos_sim_j[idx].item() for idx in role_idxs]

                    idxs = np.argsort(-np.array(role_scores))
                    if role_scores[idxs[0]] - role_scores[idxs[1]] > 1.5 * self.alpha:
                        output_i.append(role_idxs[idxs[0]])
                    else:
                        output_i.append(-1)
    
                output_list.append(output_i.copy())
        
        return output_list
    
    def change_test_ontology(self, test_ontology, test_event_types, test_role_types, tokenizer):
        self.test_event_types = test_event_types
        self.test_idx_to_type = {v:k for k,v in test_event_types.items()}
        self.test_onto = test_ontology
        self.test_role_type_idx = test_role_types # {role_name: idx}
        self.test_role_type_idx.update({"unrelated object": -1})
        self.test_rev_role_type_idx = {v:k for k,v in self.test_role_type_idx.items()}
        self.test_role_type_num = len(self.test_role_type_idx) - 1
        self.numberized_test_onto = get_numberized_onto(test_ontology, test_event_types, test_role_types)
        self.compute_test_role_reprs(tokenizer)
    
    def predict_one_example(self, tokenizer, data_item, spacy_model): 
        bert_inputs = tokenizer(data_item["sentence"], return_offsets_mapping=True)
        input_ids = torch.LongTensor([bert_inputs["input_ids"]]).to(self.device)
        attn_mask = torch.LongTensor([bert_inputs["attention_mask"]]).to(self.device)
        offset_mapping = bert_inputs["offset_mapping"]
        batch_input = {"input_ids": input_ids, "attn_mask": attn_mask}
        triggers = data_item["events"]

        entity_offsets = get_noun_entities(data_item["sentence"], spacy_model)
        seq_len = len(bert_inputs["input_ids"])

        trigger_span, entity_span = [], []
        trigger_idxs = []
        for i,trig in enumerate(triggers):
            trigger = trig["trigger"]
            for j,entity in enumerate(entity_offsets):
                trig_s, trig_e = transform_offsets(trigger[0], trigger[1], offset_mapping)
                ent_s, ent_e = transform_offsets(entity[0], entity[1], offset_mapping)
                
                trig_list = transform_to_list(trig_s+1, trig_e+1, seq_len)
                ent_list = transform_to_list(ent_s+1, ent_e+1, seq_len)

                trigger_span.append(trig_list)
                entity_span.append(ent_list)
                trigger_idxs.append(self.test_event_types[trigger[-1]])
        
        batch_input["trigger_spans"] = torch.FloatTensor([trigger_span]).to(self.device)
        batch_input["entity_spans"] = torch.FloatTensor([entity_span]).to(self.device)
        batch_input["trigger_idxs"] = torch.LongTensor([trigger_idxs]).to(self.device)
        batch_input["pair_mask"] = torch.FloatTensor([[1.0 for _ in range(len(trigger_span))]])

        # print(self.test_event_types)
        # print(self.test_role_type_idx)
        # print(self.numberized_test_onto)
        # print(batch_input["input_ids"])
        # print(batch_input["attn_mask"])
        # print(batch_input["trigger_spans"])
        # print(batch_input["entity_spans"])
        # print(batch_input["trigger_idxs"])
        # print(batch_input["pair_mask"])
        
        output_list = self.predict(batch_input)[0]
        
        output_item = deepcopy(data_item)
        for i,trigger in enumerate(triggers):
            args = []
            for j,entity in enumerate(entity_offsets):
                output_idx = i * len(entity_offsets) + j
                res = output_list[output_idx]
                if res != -1:
                    arg = [entity[0], entity[1], self.test_rev_role_type_idx[res]]
                    args.append(arg)
            output_item["events"][i].update({"arguments": args})

        return output_item


if __name__ == "__main__":
    from data import *

    with open("./data/train_ontology.json", 'r', encoding="utf-8") as f:
        train_o = json.loads(f.read())
    
    with open("./data/test_ontology.json", 'r', encoding="utf-8") as f:
        test_o = json.loads(f.read())

    train_dir = "./data/train.json"
    test_dir = "./data/test.json"

    t = BertTokenizerFast.from_pretrained("bert-large-uncased")

    train_d = ZSLDataset(train_dir, t, train_o)
    test_d = ZSLDataset(test_dir, t, test_o)

    print(train_d.role_type_idxs)
    print(test_d.role_type_idxs)

    m = ZeroShotModel("bert-large-uncased", train_d.role_type_idxs, test_d.role_type_idxs, train_o, test_o, train_d.event_type_idxs, test_d.event_type_idxs, 0.3, 1024, 256, 128, 128, 0.3, 3)
    m.to(3)
    m.compute_train_role_reprs(t)
    m.compute_test_role_reprs(t)
    
    param_groups = [
        {
            'params': [p for n, p in m.named_parameters() if n.startswith('bert')],
            'lr': 0.0001, 'weight_decay': 0.00005
        },
        {
            'params': [p for n, p in m.named_parameters() if not n.startswith('bert')],
            'lr': 0.0001, 'weight_decay': 0.00005
        }
    ] 

    print(len(param_groups[0]["params"]))  
    print(len(param_groups[1]["params"]))  

    batch_size = 3
    optimizer = AdamW(params=param_groups)
    schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=batch_size * 5, num_training_steps=batch_size * 100)

    optimizer.zero_grad()

    # for batch in DataLoader(train_d, batch_size=batch_size, shuffle=True, collate_fn=train_d.collate_fn):
    #     send_to_gpu(batch, 3)
    #     loss = m.forward(batch)
    #     print(loss)
    #     loss.backward()
    #     optimizer.step()
    #     schedule.step()
    
    i = 0
    print(m.numberized_test_onto)

    for batch in DataLoader(train_d, batch_size=batch_size, shuffle=True, collate_fn=train_d.collate_fn):
        print(torch.sum(batch["pair_mask"], 1))
        print(batch["trigger_idxs"])

        i += 1
        send_to_gpu(batch, 3)
        o = m.predict(batch)
        print(o)

        if i >= 1:
            break