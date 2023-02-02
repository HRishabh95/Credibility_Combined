import pandas as pd
import numpy as np

#TREC
# docs_100=pd.read_csv('/tmp/pycharm_project_631/docs/docs_top_100.csv',sep='\t')
# docs_100=docs_100[['qid','query','docno','text']]

docs_100=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv',sep='\t')
docs_100=docs_100[['qid','query','docno','text','rank']]
#Test_qid
# simi_score=pd.read_csv('/tmp/pycharm_project_631/result/40_60_biobert_simi_wa_d100_j10.csv',sep=' ',
#                        names=['qid','Q0','docno','rank','score','experiment'])

simi_score = pd.read_csv('/tmp/pycharm_project_631/result/40_60_biobert_simi_wa_d100_j10_trec_1M.csv',sep=' ',
                        names=['qid','Q0','docno','rank','score','experiment'])
dataset = pd.read_csv('/tmp/pycharm_project_447/test_qid.csv', sep=';')
qids=np.unique(dataset['qid'].values)

simi_score=simi_score[simi_score['qid'].isin(qids)]
simi_score.sort_values(by=['qid','rank'],inplace=True,ascending = True,)
docs_100=docs_100[docs_100['qid'].isin(qids)]
docs_100.sort_values(by=['qid','rank'],inplace=True,ascending = True,)

simi_score.to_csv('./baseline_result/wise_test.csv',sep=' ',index=None,header=None)

#CLEF 
docs_100=pd.read_csv('/tmp/pycharm_project_631/docs/clef2020_docs.csv',sep=';')
docs_100=docs_100[['qid','query','docno','text']]
simi_score=pd.read_csv('/tmp/pycharm_project_631/result/40_60_biobert_simi_wa_d100_j10_clef.csv',sep=' ',
                       names=['qid','Q0','docno','rank','score','experiment'])

dataset = pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/clef_qrels_cred.csv', sep='\t',names=['qid','Q0','docno','rel'])
qids=np.unique(dataset['qid'].values)

import random
random.seed(47)
test_set = int(len(qids) * 0.3)
test_qid = random.sample(list(qids), k=test_set)

simi_score=simi_score[simi_score['qid'].isin(test_qid)]
simi_score['experiment']='Clef_wa_d100_j10_15'
simi_score.to_csv('./extra_result/wise_test_clef_15.csv',sep=' ',index=None,header=None)