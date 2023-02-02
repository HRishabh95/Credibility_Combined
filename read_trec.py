import os
import pandas as pd
from create_result import get_averaged_score

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-1.11.0-openjdk-amd64/"

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


#Bm25 retrived
docs_100=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv',sep='\t')
docs_100=docs_100[['text','query','docno','qid']]
#qrels
qrels=pd.read_csv('/home/ubuntu/rupadhyay/2020-derived-qrels//misinfo-qrels.2aspects.useful-credible',sep=' ',
                  names=['qid','Q0','docno','t','c'])
qrels['label']=qrels['t']+qrels['c']-1

#merge
docs_merged=pd.merge(qrels,docs_100,on=['docno','qid'])
docs_merged=docs_merged[['qid','query','docno','text','label']]

simi_score=pd.read_csv('/tmp/pycharm_project_631/experiments/dtop100_jtop10/trec_1M_similarity_biobert.csv',
                       sep='\t')
# simi_score=pd.read_csv('/tmp/pycharm_project_631/result/40_60_biobert_simi_wa_d100_j10_trec_1M.csv',
#                        sep='\t')

average_score=get_averaged_score(simi_score)
docs_merged_score=pd.merge(docs_merged,average_score,on=['docno','qid'])

docs_merged_score.to_csv('./datset.csv',sep=';',index=None)