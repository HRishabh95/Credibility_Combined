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
from utils import mkdir_p

cred_score=True

data='trec'

if data=='trec':
    #TREC
    docs_100=pd.read_csv('/tmp/pycharm_project_631/docs/docs_top_100.csv',sep='\t')
    docs_100=docs_100[['qid','query','docno','text','score']]
    simi_score=pd.read_csv('./similarity_score_sw_biobert.csv',
                           sep='\t')
    # docs_100 = pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv', sep='\t')
    # docs_100 = docs_100[['text', 'query', 'docno', 'qid']]
    # simi_score = pd.read_csv('/tmp/pycharm_project_631/experiments/dtop100_jtop10/trec_1M_similarity_biobert.csv',
    #                          sep='\t')
    average_score=get_averaged_score(simi_score)
    qrels_file = "/home/ubuntu/rupadhyay/2020-derived-qrels/misinfo-qrels.cred.wise"
    dataset = pd.read_csv(qrels_file, sep=' ',names=['qid','Q0','docno','label'])
    qids=np.unique(dataset['qid'].values)
    average_score=average_score[average_score['qid'].isin(qids)]
    dataset=pd.merge(average_score,docs_100,on=['qid','docno'])


#CLEF

if data=='clef':
    docs_100=pd.read_csv('/tmp/pycharm_project_631/docs/clef2020_docs.csv',sep=';')
    docs_100=docs_100[['qid','query','docno','text','score']]
    simi_score=pd.read_csv('/tmp/pycharm_project_631/experiments/dtop100_jtop10/clef2020_similarity_biobert.csv',
                           sep='\t')

    average_score=get_averaged_score(simi_score)

    dataset=pd.merge(average_score,docs_100,on=['qid','docno'])
#8 , 24


for batch_number,batch in enumerate([2]):
    for epoch_number, epoch in enumerate([6]):
        print(batch,epoch)
        if cred_score:
            model_path=f'''./cross_encoder_CRRerank_bert-base-uncased_v2_clef/cross_encoder_{epoch}_{batch}_bm25/'''
            file_path=f'''./cross_encoder_CRRerank_bert-base-uncased_v2_clef/results_evaluation/cross_encoder_bert_{epoch}_{batch}_bm25.csv'''
            mkdir_p("/".join(file_path.split("/")[:-1]))
        else:
            model_path=f'''./cross_encoder_CRRerank_biobert-v1.1_clef/cross_encoder_{epoch}_{batch}/'''
            file_path = f'''./cross_encoder_CRRerank_biobert-v1.1_clef/results_evaluation/cross_encoder_biobert_{epoch}_{batch}.csv'''
        if not os.path.isfile(file_path):
            if os.path.isfile(model_path+"/config.json"):
                model = CrossEncoder(model_path, num_labels=1, max_length=510)

                result_df=[]
                for ii,row in tqdm(dataset.iterrows()):
                    if cred_score:
                        # score = [r for r in list(str(row['c_score'])) if r != '.']
                        # c_score = f'''{row['c_score']:.4f} [SEP]'''
                        # c_score= f'''topical score is {row['score']:.0f} '''
                        # c_score=f'''credibility score of the document is {float(row['c_score']):.4f} and topical score is {float(row['score']):.0f}'''
                        c_score = f'''{row['score']:.1f} [SEP]'''  # Acc:0.7, F1- 0.69

                    else:
                        c_score=''
                    score=model.predict([row['query'],c_score+row['text']])
                    result_df.append([row['qid'],0,row['docno'],score])

                result_df=pd.DataFrame(result_df,columns=['qid','Q0','docno','score'])
                qids = np.unique(result_df.qid.values)
                sorted_dfs=[]
                for qid in qids:
                    if qid!=28 and qid!=33:
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
                    result_df['experiment']=f'''cross_encoder_bert_{epoch}_{batch}_bm25'''
                else:
                    result_df['experiment'] = f'''cross_encoder_biobert_{epoch}_{batch}'''

                result_df.to_csv(file_path,sep=' ',index=None,header=None)
