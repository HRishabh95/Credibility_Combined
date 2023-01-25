import datasets
import pandas as pd
import torch
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

cred_score=False
if cred_score:
    model_path='/home/ricky/PycharmProjects/BERT_TC_train/cross_encoder_result/cross_encoder_7_2c_score'
    file_path='./results/cross_encoder_binary_c_score.csv'
else:
    model_path='/home/ricky/PycharmProjects/BERT_TC_train/cross_encoder_result/cross_encoder_7_2'
    file_path = './results/cross_encoder_binary.csv'

dataset=pd.read_csv('./datset.csv',sep=';')
model = CrossEncoder(model_path, num_labels=1, max_length=510)

result_df=[]
for ii,row in tqdm(dataset.iterrows()):
    #c_score= f'''The credibility score of the document is {str(row['c_score'])}. '''
    c_score=''
    score=model.predict([row['query'],c_score+row['text']])
    result_df.append([row['qid'],0,row['docno'],score])

result_df=pd.DataFrame(result_df,columns=['qid','Q0','docno','score'])
qids = np.unique(result_df.qid.values)
sorted_dfs=[]
for qid in qids:
    if qid!=28:
        qid_df=result_df.loc[result_df['qid']==qid]
        sorted_qid_df=qid_df.sort_values('score',ascending=False).reset_index()
        sorted_qid_df['n_rank']=1
        for i in sorted_qid_df.index:
            sorted_qid_df.at[i,'n_rank']=i+1
        sorted_dfs.append(sorted_qid_df)

sorted_qid_df_concat=pd.concat(sorted_dfs)
result_df=sorted_qid_df_concat[['qid','Q0','docno','n_rank','score']]
result_df.columns=['qid','Q0','docno','rank','score']
result_df['experiment']='cross_encoder'

result_df.to_csv(file_path,sep=' ',index=None,header=None)
