# Zero-shot Event Argument Extraction
This repo contains the source code for training and testing for the event argument extraction (EAE) task in a zero-shot manner. In general, we built a shared embedding space for both mentions in a sentence and type names in the ontology. Our system only requires the event argument names like `Attacker`, `Agent` and `Victim` to conduct zero-shot training and testing on argument role types in all scenarios.

## Overview of Methodology
Our zero-shot event argument extraction model only requires the event argument role names (usually single words or phrases) for each event type (e.g., the event argument role names `Giver`, `Beneficiary`, `Recipient` and `Place` for event type `Transaction: Transfer-Money`).
Our model does not require any detailed information such as natural language descriptions, example annotations or external training corpus.
Our basic idea is to build a shared embedding space for both role label semantics and the contextual text features between triggers and arguments.
Given an input sentence, we first perform named entity recognition (NER) with Spacy to extract all entity mentions in a sentence.
After that, given the event role names for a certain event type, we first obtain the semantic embeddings using the pretrained language model `bert-large-cased`.
Then we also use BERT to get the representation vectors for all extracted event triggers and entity mentions within the sentence, and concatenate the vectors as to represent a trigger-entity pair.
The intuition here is to learn two separate neural network projection functions to map each role label and trigger-entity pair into a single shared embedding space, where each trigger-entity pair stays near its correct roles and far from all other event argument roles.
During training, we minimize the cosine distance between each trigger-entity pair with its corresponding role label, while maximizing the distance between the trigger-entity pair and all other negative role labels.
In this way, the trigger-entity pair representations tend to be centered around their argument role labels.
During testing, we directly classify each trigger-entity pair as its nearest role label. 

## Installation
Please use `pip install -r requirements.txt` to install python dependencies.

## File Structure
+ `./checkpoints`: stores the trained model checkpoints
+ `./data`: stores training and testing data (from ACE-05)
+ `./ontology`: stores the training and testing types definition in `.json` format.
+ `data.py`: implementation of dataset and dataloader objects for training neural network models.
+ `model.py`: the core implementation of zero-shot event argument extraction model.
+ `eval.py`: Python script for evaluation and computing scores.
+ `predict.py`: Heuristic entity extraction for predicting the arguments.
+ `utils.py`: Utility functions like canculating averaged embeddings for each word span.

## Usage
Please run the `train.py` for training and testing zero-shot event argument extraction models. Important command line arguments:
+ `--epochs`: maximum number of training epochs.
+ `--batch`: training batch size.
+ `--alpha`: hyperparameter for negative loss.
+ `--lr`: learning rate.
+ `--gpu`: GPU device id to train the model on.
+ `--train_types`: the file name defining the training ontology, e.g., `ace_train_10` denotes the ontology file at `./ontology/ace_train_10.json`.
+ `--test_types`: the file name defining the testing ontology, e.g., `ace_test_23` denotes the ontology file at `./ontology/ace_test_23.json`.

+ `--train_data`: the file name defining the training data, e.g., `train` denotes the ontology file at `./data/train.json`.
+ `--test_data`: the file name defining the testing data, e.g., `test` denotes the ontology file at `./data/test.json`.

+ `--eval`: if this argument is set, the model will only run evaluation on `--save_dir`.
+ `--save_dir`: model save directory, defaultly set as `./checkpoints`.
+ `--name`: the name of this experiment.

### Example usage:
`python train.py --name RUN_NAME --train_types ace_train_10 --test_types ace_test_23 --gpu 0`
