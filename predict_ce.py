import os.path
import datasets
import pandas as pd
import torch
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from create_result import get_averaged_score

docs_100=pd.read_csv('/tmp/pycharm_project_631/docs/docs_top_100.csv',sep='\t')
docs_100=docs_100[['qid','query','docno','text']]

simi_score=pd.read_csv('./similarity_score_sw_biobert.csv',
                       sep='\t')

average_score=get_averaged_score(simi_score)

cred_score=True

dataset = pd.read_csv('/tmp/pycharm_project_447/test_qid.csv', sep=';')
qids=np.unique(dataset['qid'].values)

average_score=average_score[average_score['qid'].isin(qids)]
dataset=pd.merge(average_score,docs_100,on=['qid','docno'])


#8 , 24
for batch in [2,4,6,8,10]:
    for epoch in [6,7,8,9]:
        print(batch,epoch)
        if cred_score:
            model_path=f'''/tmp/pycharm_project_447/cross_encoder_CRRerank_bert-large-uncased/cross_encoder_{epoch}_{batch}_c_score/'''
            file_path=f'''./test_results_large/cross_encoder_bert_base_{epoch}_{batch}_c_score.csv'''
        else:
            model_path = f'''/tmp/pycharm_project_447/cross_encoder_CRRerank_bert-large-uncased/cross_encoder_{epoch}_{batch}/'''
            file_path = f'''./test_results_large/cross_encoder_bert_base_{epoch}_{batch}.csv'''
        if not os.path.isfile(file_path):
            if os.path.isfile(model_path+"/config.json"):
                model = CrossEncoder(model_path, num_labels=1, max_length=510)

                result_df=[]
                for ii,row in tqdm(dataset.iterrows()):
                    if cred_score:
                        c_score= f'''The credibility score of the document is {str(row['c_score'])}. '''
                    else:
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
                if cred_score:
                    result_df['experiment']=f'''cross_encoder_bert_large_{epoch}_{batch}_c_score'''
                else:
                    result_df['experiment'] = f'''cross_encoder_bert_large_{epoch}_{batch}'''

                result_df.to_csv(file_path,sep=' ',index=None,header=None)
