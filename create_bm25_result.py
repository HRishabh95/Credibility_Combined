import pandas as pd

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
  text = re.sub(r"[^A-Za-z0-9(),.!?\'`]", " ", text)
  #text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

def trec_generate(f):
  df = pd.DataFrame(f, columns=['docno', 'text'])
  return df

#TREC
#Bm25 retrived
docs_100=pd.read_csv('/tmp/pycharm_project_631/docs/docs_top_100.csv',sep='\t')
docs_100=docs_100[['qid','query','docno','rank','score']]

# docs_100=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv',sep='\t')
# docs_100=docs_100[['qid','query','docno','rank','score']]
#Test_qid
test_data=pd.read_csv('/tmp/pycharm_project_447/test_qid.csv',sep=';')
test_data=test_data[['qid','query','docno','Q0']]

qids=np.unique(test_data['qid'].values)

docs_100=docs_100[docs_100['qid'].isin(qids)]
#merge
# final_result=pd.merge(test_data,docs_100,on=['qid','docno'])
docs_100.sort_values(by=['qid','rank'],ascending = True,
               inplace = True,)

docs_100['Q0']=0
docs_100=docs_100[['qid','Q0','docno','rank','score']]
docs_100['experiment']='bm25_100k'


docs_100.to_csv('/tmp/pycharm_project_447/baseline_result/test_data.csv',sep=' ',index=None,header=None)



##CLEF
#Bm25 retrived
docs_100=pd.read_csv('/tmp/pycharm_project_631/docs/clef2020_docs.csv',sep=';')
docs_100=docs_100[['qid','docno','rank','score']]

# test_data = pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/clef_qrels_cred.v2.csv', sep='\t',names=['qid','Q0','docno','rel'])
# #merge
# final_result=pd.merge(test_data,docs_100,on=['qid','docno'])
docs_100.sort_values(by=['qid','rank'],ascending = True,
               inplace = True,)

docs_100['Q0']=0
docs_100=docs_100[['qid','Q0','docno','rank','score']]
docs_100['experiment']='bm25_clef_15'
import random
random.seed(47)
test_set = int(len(qids) * 0.3)
test_qid = random.sample(list(qids), k=test_set)

docs_100=docs_100[docs_100['qid'].isin(test_qid)]

docs_100.to_csv('./extra_result/clef_bm25_data_15.csv',sep=' ',index=None,header=None)
