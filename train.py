import torch
import torch.nn as nn
import argparse
import tqdm
import os
import spacy
import nltk

from data import *
from eval import *
from utils import send_to_gpu
from model import ZeroShotModel
from transformers import BertTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

# Hyper Params
torch.cuda.manual_seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=24)
parser.add_argument('--bert_dim', type=int, default=1024)
parser.add_argument('--tdim', type=int, default=256, help="hidden dim for the textual encoder")
parser.add_argument('--rdim', type=int, default=128, help="hidden dim for the role name encoder")
parser.add_argument('--out_dim', type=int, default=128, help="dimension for the projected space")
parser.add_argument('--alpha', type=float, default=0.1, help="ratio for negative loss")
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--decay', type=float, default=0.00005)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--gpu', type=int, default=3)

parser.add_argument('--train_types', type=str, default="ace_train_10")
parser.add_argument('--test_types', type=str, default="ace_test_23")

parser.add_argument('--train_data', type=str, default="train")
parser.add_argument('--test_data', type=str, default="test")

parser.add_argument('--eval', dest='eval', action="store_true", help='evaluate on gpu or cpu.')
parser.add_argument('--save_dir', type=str, default="./checkpoints/")
parser.add_argument('--name', type=str, default="default")
args = parser.parse_args()

model_dir = os.path.join(args.save_dir, args.name)

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# args.gpu = "cpu"
# read in train and test ontology
train_ontology_file = os.path.join("./ontology", args.train_types + ".json")
test_ontology_file = os.path.join("./ontology", args.test_types + ".json")

train_data_file = os.path.join("./data", args.train_data + ".json")
test_data_file = os.path.join("./data", args.test_data + ".json")

with open(train_ontology_file, 'r', encoding="utf-8") as f:
    train_o = json.loads(f.read())
    
with open(test_ontology_file, 'r', encoding="utf-8") as f:
    test_o = json.loads(f.read())

# read bert tokenizer
t = BertTokenizerFast.from_pretrained("bert-large-uncased")

# read dataset
train_d = ZSLDataset(train_data_file, t, train_o)
test_d = ZSLDataset(test_data_file, t, test_o)

m = ZeroShotModel("bert-large-uncased", train_d.role_type_idxs, test_d.role_type_idxs, train_o, test_o, train_d.event_type_idxs, test_d.event_type_idxs, args.dropout, args.bert_dim, args.tdim, args.rdim, args.out_dim, args.alpha, args.gpu)
m.to(args.gpu)
# compute test and train initial role type embeddings
m.compute_train_role_reprs(t)
m.compute_test_role_reprs(t)
# print(m.test_role_reprs[m.test_role_type_idx["Beneficiary"]])
# print(m.test_role_reprs[:, 0])
# print(m.test_role_reprs[m.test_role_type_idx["Beneficiary"]])
# print(m.test_role_reprs[m.test_role_type_idx["Attacker"]])
# print(m.test_role_reprs[m.test_role_type_idx["Place"]])


# batch_size and training
batch_size = args.batch
num_epochs = args.epochs
lrate = args.lr
weight_decay = args.decay
warmup = args.warmup

param_groups = [
    {
        'params': [p for n, p in m.named_parameters() if n.startswith('bert')],
        'lr': lrate, 'weight_decay': weight_decay
    },
    {
        'params': [p for n, p in m.named_parameters() if not n.startswith('bert')],
        'lr': lrate, 'weight_decay': weight_decay
    }
] 

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=batch_size * warmup, num_training_steps=batch_size * num_epochs)

# Training:
batch_num = len(train_d) // batch_size
num_steps = 0

if not args.eval:
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))
        m.train()
        progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))

        optimizer.zero_grad()
        train_loss = 0.0

        # training loop
        for batch_idx, batch in enumerate(DataLoader(train_d, batch_size=batch_size, shuffle=True, collate_fn=train_d.collate_fn)):
            send_to_gpu(batch, args.gpu)
            loss = m(batch)
            loss.backward()
            train_loss += loss.item()

            progress.update(1)
            nn.utils.clip_grad_norm_(m.parameters(), args.grad_clip)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

        progress.close()
        print("Training Loss: --- " + str(train_loss / (batch_idx + 1)))
        
        # testing loop
        m.eval()
        output_results = []
        output_labels = []

        best_f1 = 0.000
        for batch_idx, batch in enumerate(DataLoader(test_d, batch_size=batch_size, shuffle=False, collate_fn=test_d.collate_fn)):
            send_to_gpu(batch, args.gpu)
            pred_res = m.predict(batch)
            gold_res = batch["gold_labels"]

            output_results.extend(pred_res.copy())
            output_labels.extend(gold_res.copy())

        stats, overall_res = evaluate(output_results, output_labels, test_d.role_type_idxs)
        print(stats)
        print(overall_res)

        if overall_res[-1] > best_f1:
            torch.save(m.state_dict(), model_dir+"/checkpoint.pt")
            num_steps = 0
            best_f1 = overall_res[-1]
        else:
            num_steps += 1
        
        if num_steps > 5:
            print("Training Ends.")
            break

else:
    m.load_state_dict(torch.load(model_dir+"/checkpoint.pt"))
    m.eval()
    # output_results = []
    # output_labels = []

    # best_f1 = 0.000
    # for batch_idx, batch in enumerate(DataLoader(test_d, batch_size=batch_size, shuffle=False, collate_fn=test_d.collate_fn)):
    #     send_to_gpu(batch, args.gpu)
    #     pred_res = m.predict(batch)
    #     gold_res = batch["gold_labels"]

    #     output_results.extend(pred_res.copy())
    #     output_labels.extend(gold_res.copy())

    # stats, overall_res = evaluate(output_results, output_labels, test_d.role_type_idxs)
    # print(stats)
    # print(overall_res)

    spacy_model = spacy.load("en_core_web_sm")
    tt = nltk.tokenize.TreebankWordTokenizer()

    input_data = {"sentence":"This means that the government of one of the poorest countries in the world was paying Enron 220 million US dollars a year!", "events":[{"trigger": [80, 86, "Transaction:Transfer-Money"]}]}
    m.predict_one_example(t, input_data, tt)






