from data import ZSLDataset

import spacy
import nltk
from nltk.tokenize import TreebankWordTokenizer

# def prepare
def get_spacy_entities(sent, model):
    entity_type_list = ["ORG", "PERSON", "GPE", "LOC"]
    token_offsets = []
    
    doc = model(sent)
    for ent in doc.ents:
        # print(ent.text)
        # print(ent.label_)
        # print(ent.start_char)
        # print(ent.end_char)
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


if __name__ == "__main__":
    s = "This means that the government of one of the poorest countries in the world was paying Enron 220 million US dollars a year not to produce electricity!"
    # s = "John hits the United States."

    nlp = spacy.load("en_core_web_sm")
    t = TreebankWordTokenizer()

    entities = get_spacy_entities(s, nlp)
    print(entities)

    entities = get_noun_entities(s, t)
    print(entities)
