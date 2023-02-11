from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
# model_name='/tmp/pycharm_project_447/cross_encoder_CRRerank_biobert-v1.1/cross_encoder_2_12_c_score_deci_4_mrr'
model_name= '/tmp/pycharm_project_447/cross_encoder_CRRerank_biobert-v1.1_clef/cross_encoder_2_16_c_score_original/'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True,max_length=510,truncation=True,padding=True,add_special_tokens=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)

docs_100=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv',sep='\t')
docs_100=docs_100[['qid','query','docno','text','rank','score']]
docs_100.colums=['qid','query','docno','text','rank','bm25']
#Test_qid
simi_score=pd.read_csv('/tmp/pycharm_project_631/result/40_60_biobert_simi_wa_d100_j10.csv',sep=' ',
                       names=['qid','Q0','docno','rank','score','experiment'])

# dataset = pd.read_csv('/tmp/pycharm_project_447/test_qid.csv', sep=';')
# qids=np.unique(dataset['qid'].values)
#
# simi_score=simi_score[simi_score['qid'].isin(qids)]
simi_score=simi_score.merge(docs_100,on=['qid','docno'])

import shap
import transformers
import nlp
import torch
import numpy as np
import scipy as sp

# masker=shap.maskers.Text(tokenizer=r"\W")

masker=shap.maskers.Text(tokenizer=tokenizer)

label_names = ["Retrieve", "Non Relevent"]


bio_bert_result=pd.read_csv('/tmp/pycharm_project_447/cross_encoder_CRRerank_biobert-v1.1_clef/results_trec/cross_encoder_biobert_2_16_c_score_original.csv',sep=' ',
                            names=['qid','Q0','docno','rank','score','experiment'])

#3,37, 41
explainer = shap.Explainer(pred,masker=masker,seed=47)
row=simi_score[(simi_score['docno']=='63505854-7664-462e-b2ea-852644c5ec02') & (simi_score['qid']==13)]
c_score = f'''credibility score of the document is {0.2+row['score_x'].values[0]:.4f}'''
row['text']="The city is however to record a Covid-19 case but residents are already taking measures to curb its spread hence the increasing use of face masks by many residents. Many believe wearing masks protects an individual from contracting the deadly virus that has killed thousands of people worldwide. There are a number of stories circulating on social media urging people to wear masks to protect themselves from Covid-19. Health specialist Dr Prince Murambi said the move by members of the public to wear the masks even if they are not infected, is good because it reduces chances of spreading the virus in the event one wearing it is infected. The surgical mask which is being used by most people prevents an individual from spreading respiratory droplets when coughing or sneezing. We are therefore encouraging people to wear them when in public places so that in the event they have the virus they do not pass it on to other people,he said. Dr Murabmi said the other mask is a disposable N95 which protects an individual wearing it from breathing in some hazardous substances. An N95 mask is convenient in that it protects people from breathing in small particles in the air such as dust. It is designed to filter out at least 95% of the dust in the air hence somehow protect people from contracting the Covid-19,he said. Dr Murambi said a mask should just be used once and discarded. According to the World Health Organisation individuals taking care of persons suspected to be infected by the virus should wear face masks. Masks are however effective when used in combination with frequent hand-cleaning with alcohol-based hand rub or soap and water. Some youths in some eastern suburbs of Bulawayo were seen donning fashionable masks. The youths seem to be following in the footsteps of stylish visitors at the London Fashion week who rocked up in dapper masks regarded as coral masks. Lucky for me, I’ve been to parties which had themes around masks so I simply took one from the wardrobe and now use it to protect myself from coronavirus.I feel safe when wearing it,said Louise Charambira from Hillside suburb. Asked if such masks were advisable to wear for that purpose, Harare-based Fortress Hospital Administrator Ms Sibusisiwe Dube said: That’s totally not ideal because you never know if there’s an infection or not on the cloth. Members of the public should therefore desist from using such masks.Ms Dube said she had observed that many people had no knowledge on the wearing and removing of masks. The way people are removing their masks, putting them around the neck and then wearing them again puts them at risk of contracting diseases. People should wear the mask as and when necessary,said Ms Dube. She said once removed, the mask should be replaced by a new one. Dr Nyasha Masuka said people should follow the guidelines on how to use and wear masks and the correct masks to use. Mpilo Central Hospital Clinical Director Dr Solwayo Ngwenya said infected people can stop the spread of the virus if they wear masks."
example=f'''{row['query'].values[0]} [SEP] {c_score} {clean_en_text(row['text'].values[0])}'''

# row=simi_score.iloc[369]
# c_score = f'''The credibility score of the document is {row['c_score']:.4f}'''
# example_2=f'''{row['query']} [SEP] {c_score} {clean_en_text(row['text'])}'''

shap_values= explainer([" ".join(example.split(" ")[:400])])

file = open('qid_%s_docno.html'%row['qid'].values[0],'w')
file.write(shap.plots.text(shap_values,display=False))
file.close()


# shap.plots.bar(shap_values[:,:, 'LABEL_0'].mean(axis=0), max_display=30,order=shap.Explanation.argsort)

