from dataset_IG import HMDataset

from smooth_gradient import SmoothGradient
from integrated_gradient import IntegratedGradient


import torch
from torch import nn
from torch.utils.data import DataLoader


from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoConfig
import transformers
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
import random
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


import os
import pandas as pd
import numpy as np


import re
import string
PUNCTUATIONS = string.punctuation.replace(".",'')


def remove_punctuation(text):
  trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
  return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
  """
  text
  """
  text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)
  #text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()


# Load pre-trained model tokenizer
model_name='/tmp/pycharm_project_447/cross_encoder_CRRerank_biobert-v1.1/cross_encoder_2_12_c_score_deci_4_mrr'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True,max_length=510,truncation=True,padding=True,add_special_tokens=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)

docs_100=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv',sep='\t')
docs_100=docs_100[['qid','query','docno','text','rank']]
#Test_qid
simi_score=pd.read_csv('/tmp/pycharm_project_631/result/40_60_biobert_simi_wa_d100_j10.csv',sep=' ',
                       names=['qid','Q0','docno','rank','score','experiment'])

dataset = pd.read_csv('/tmp/pycharm_project_447/test_qid.csv', sep=';')
qids=np.unique(dataset['qid'].values)

simi_score=simi_score[simi_score['qid'].isin(qids)]
simi_score=simi_score.merge(dataset,on=['qid','docno'])

import shap
import transformers
import nlp
import torch
import numpy as np
import scipy as sp


# explainer = shap.Explainer(pred,tokenizer,seed=47)

label_names = ["Retrieve", "Non Relevent"]



row=simi_score.iloc[204]
c_score = f'''credibility score of the document is {row['c_score']:.4f}'''
example=f'''{row['query']} [SEP] {c_score} {clean_en_text(row['text'])}'''

row=simi_score.iloc[369]
c_score = f'''credibility score of the document is {row['c_score']:.4f}'''
example_2=f'''{row['query']} [SEP] {c_score} {clean_en_text(row['text'])}'''


test_example=[example,example_2]


test_dataset = HMDataset(
    data_list=test_example,
    tokenizer=tokenizer,
    max_length=510,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
)

criterion = nn.CrossEntropyLoss()

integrated_grad = IntegratedGradient(
    model,
    criterion,
    tokenizer,
    show_progress=False,
    encoder="bert"
)


instances = integrated_grad.saliency_interpret(test_dataloader)
