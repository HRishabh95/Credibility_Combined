
import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import models, SentenceTransformer,util
cred_score=True

if cred_score:
    model_path='/home/ricky/PycharmProjects/BERT_TC_train/sent_MNR_models/sbert_test_MultipleNegativesRankingLoss_10_3c_score'
    file_path='./results/sent_transform_c_score.csv'
else:
    model_path='/home/ricky/PycharmProjects/BERT_TC_train/sent_MNR_models/sbert_test_MultipleNegativesRankingLoss_10_3'
    file_path = './results/sent_transformer.csv'

bert = models.Transformer(model_path)
pooler = models.Pooling(
    bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
dataset=pd.read_csv('./datset.csv',sep=';')
model = SentenceTransformer(modules=[bert, pooler])
result_df=[]
for ii,sample in tqdm(dataset.iterrows()):
    if cred_score:
        c_score = f'''The credibility score of the document is {str(sample['c_score'])}. '''
    else:
        c_score = ''

    embeddings1 = model.encode(sample['query'], convert_to_tensor=True)
    embeddings2 = model.encode(c_score + sample['text'], convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    result_df.append([sample['qid'],0,sample['docno'],cosine_scores.tolist()[0][0]])

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
result_df['experiment']='sen_transformer'

result_df.to_csv(file_path,sep=' ',index=None,header=None)
