import torch
from torch.nn.functional import cross_entropy
from transformers import BertTokenizer
import pandas as pd
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
docs_merged_score= pd.read_csv('./datset.csv',sep=';')
pos_neg_dataset=[]
no_qid={}
qids=docs_merged_score['qid'].unique()
for qid in qids:
    s = ''
    if qid not in no_qid:
        no_qid[qid]=0
    for ii,rows in docs_merged_score[docs_merged_score['qid']==qid].iterrows():
        if rows['label']==0 and s!='neg':
            pos_neg_dataset.append(rows.tolist())
            s='neg'
            no_qid[qid]+=1
        elif rows['label']==1 and s!='pos':
            pos_neg_dataset.append(rows.tolist())
            s='pos'
            no_qid[qid]+=1

pos_neg_dataset=pd.DataFrame(pos_neg_dataset,columns=docs_merged_score.columns)
pos_neg_dataset['c_score']=pos_neg_dataset['c_score'].astype(str)
qids=[]
for r,j in no_qid.items():
    if j>=10:
        qids.append(r)

final_dataset_train=[]
final_dataset_test=[]
for qid in qids:
    final_dataset_train.append(pos_neg_dataset[pos_neg_dataset['qid']==qid].head(6))
    final_dataset_test.append(pos_neg_dataset[pos_neg_dataset['qid']==qid].tail(4))

final_dataset_train=pd.concat(final_dataset_train).to_csv('./train.csv',sep=';')
final_dataset_test=pd.concat(final_dataset_test).to_csv('./test.csv',sep=';')

final_dataset_train=json.loads(pd.concat(final_dataset_train).to_json(orient='records'))
final_dataset_test=json.loads(pd.concat(final_dataset_test).to_json(orient='records'))

with open("train.jsonl", "w") as f:
    for item in final_dataset_train:
        f.write(json.dumps(item) + "\n")

with open("test.jsonl", "w") as f:
    for item in final_dataset_test:
        f.write(json.dumps(item) + "\n")

## train


from transformers import RobertaTokenizer,RobertaForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def preprocess_function(batch):
    return tokenizer(batch["query"], batch["text"],batch['c_score'],truncation=True, padding="max_length")

import datasets
dataset = datasets.load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})

tokenized_data = dataset.map(preprocess_function, batched=True,batch_size=2)

from transformers import Trainer, TrainingArguments
import numpy as np

def compute_metrics(eval_pred):
    f1_score = datasets.load_metric("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_score.add_batch(predictions=predictions, references=labels)
    return f1_score.compute()


model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=3,  # total # of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    learning_rate=2e-5,  # learning rate
    save_total_limit=2,  # limit the total amount of checkpoints, delete the older checkpoints
    logging_dir="./logs",  # directory for storing logs
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=tokenized_data["train"],  # training dataset
    eval_dataset=tokenized_data["test"],  # evaluation dataset
    compute_metrics=compute_metrics,  # the callback that computes metrics of interest
)

trainer.train()
trainer.evaluate()
