from transformers import BertModel, BertTokenizerFast

import torch
import torch.nn as nn
import numpy as np

import spacy
import nltk
from nltk.tokenize import TreebankWordTokenizer
from copy import deepcopy


def get_spacy_entities(sent, model):
    entity_type_list = ["ORG", "PERSON", "GPE", "LOC"]
    token_offsets = []
    
    doc = model(sent)
    for ent in doc.ents:
        if ent.label_ in entity_type_list:
            token_offsets.append([ent.start_char, ent.end_char])

    return token_offsets


def get_noun_entities(sent, nltk_tokenizer):
    offsets = []
    tokens = nltk_tokenizer.tokenize(sent)
    spans = list(nltk_tokenizer.span_tokenize(sent))
    pos_tags = nltk.pos_tag(tokens)
    pos_tag_num = len(pos_tags)

    pos_tags.append(("null", "null"))

    for i in range(pos_tag_num):
        if pos_tags[i][1].startswith("NN") and (not pos_tags[i+1][1].startswith("NN")):
            offsets.append(spans[i])
    return offsets  

def padding_mask_list(input_mask_list):
    mask_list = deepcopy(input_mask_list)
    max_subwords_num = max([len(mask) for mask in mask_list])
    for mask in mask_list:
        mask[0] = 0
        mask[-1] = 0
    padded_attn_mask_list = [mask+[0 for _ in range(max_subwords_num-len(mask))] for mask in mask_list]
    return padded_attn_mask_list

def pw_cosine_similarity(input_a, input_b):
    normalized_input_a = nn.functional.normalize(input_a)  
    normalized_input_b = nn.functional.normalize(input_b)
    res = torch.mm(normalized_input_a, normalized_input_b.T)
    return res


def get_bert_embeddings(word_list, model, tokenizer, gpu):
    # input: a list of words, bert_model, bert_tokenizer
    # output: numpy tensor (word_num, dim)
    segments = tokenizer(word_list)
    attn_mask_list = segments["attention_mask"]
    padded_attn_mask_list = padding_mask_list(attn_mask_list)
    padded_segments = tokenizer.pad(segments)
    input_ids, attn_mask = padded_segments["input_ids"], padded_segments["attention_mask"]

    if gpu == "cpu":
        batch_input_ids, batch_attn_mask = torch.LongTensor(input_ids), torch.LongTensor(attn_mask)
        batch_padded_mask = torch.FloatTensor(padded_attn_mask_list)
    else:
        batch_input_ids, batch_attn_mask = torch.LongTensor(input_ids).to(gpu), torch.LongTensor(attn_mask).to(gpu)
        batch_padded_mask = torch.FloatTensor(padded_attn_mask_list).to(gpu)
    
    encodes = model(batch_input_ids, attention_mask=batch_attn_mask)[0]

    avg_padded_mask = batch_padded_mask / (torch.sum(batch_padded_mask, 1).unsqueeze(-1))
    output_embeds = torch.sum(torch.stack([avg_padded_mask for _ in range(encodes.shape[-1])], 2) * encodes, 1)
    return output_embeds


def send_to_gpu(batch, gpu):
    for key,item in batch.items():
        if hasattr(item, "shape"):
            batch[key] = item.to(gpu)



if __name__ == "__main__":
    b = BertModel.from_pretrained("bert-large-uncased")
    t = BertTokenizerFast.from_pretrained("bert-large-uncased")

    roles = ['Adjudicator', 'Agent', 'Artifact', 'Attacker', 'Beneficiary', 'Buyer', 'Defendant', 'Destination', 'Entity', 'Giver', 'Instrument', 'Organization', 'Origin', 'Person', 'Place', 'Plaintiff', 'Prosecutor', 'Recipient', 'Seller', 'Target', 'Vehicle', 'Victim']
    words = [role.lower() for role in roles]
    # for word in words:
    #     print(t.tokenize(word))

    word_embeds = get_bert_embeddings(words, b, t, "cpu")
    # cos_sim = pw_cosine_similarity(word_embeds, word_embeds)
    # # print(cos_sim)
    # role_num = len(roles)

    # for i in range(role_num):
    #     for j in range(role_num):
    #         print(roles[i], end=" ")
    #         print(roles[j], end=" ")
    #         print(cos_sim[i][j])
    # # a = torch.rand(4,5)
    # b = torch.rand(5,5)
    # c = pw_cosine_similarity(a, b)
    # print(c)