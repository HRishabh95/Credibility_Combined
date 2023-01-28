from trectools import TrecQrel, procedures
from scipy.stats import hmean
import pandas as pd
import numpy as np

def get_mrr(qrels_file,run_file,top_n=10):
    qrels=pd.read_csv(qrels_file,sep=' ',names=['qid','Q0','docno','grade'])
    runs=pd.read_csv(run_file,sep=' ',index_col=None,header=None,names=['qid','Q0','docno','rank','score','experiment'])
    if runs['rank'].min()==0:
        runs['rank']=runs['rank']+1
    runs=runs[runs['rank']<top_n]
    runs['docno']=runs['docno'].astype('object')
    runs['qid']=runs['qid'].astype(int)
    merged_pd=pd.merge(qrels,runs,on=['qid','docno'],how='inner')
    relevances_rank = merged_pd.groupby(['qid', 'grade'])['rank'].min()
    ranks = relevances_rank.loc[:, 1]
    reciprocal_ranks = 1 / (ranks)
    return  reciprocal_ranks.mean()


def get_ndcg_p(qrels_file):
    qrels = TrecQrel(qrels_file)
    runs=procedures.list_of_runs_from_path('./extra_result/','*.csv')

    results = procedures.evaluate_runs(runs, qrels, per_query=True)
    top_NDCG_10,top_NDCG_20,top_NDCG_30,top_P_10,top_P_20,top_P_30,top_mrr,top_map,run_id=[],[],[],[],[],[],[],[],[]
    for result in results:
        run_id.append(result.runid)
        result=result.data
        top_NDCG_10.append(result.loc[(result['metric']=='NDCG_10')&(result['query']=='all')]['value'].values[0])
        top_NDCG_20.append(result.loc[(result['metric'] == 'NDCG_20') & (result['query'] == 'all')]['value'].values[0])
        top_NDCG_30.append(result.loc[(result['metric'] == 'NDCG_30') & (result['query'] == 'all')]['value'].values[0])
        top_mrr.append(result.loc[(result['metric']=='recip_rank')&(result['query']=='all')]['value'].values[0])
        top_P_10.append(result.loc[(result['metric']=='P_10')&(result['query']=='all')]['value'].values[0])
        top_P_20.append(result.loc[(result['metric'] == 'P_20') & (result['query'] == 'all')]['value'].values[0])
        top_P_30.append(result.loc[(result['metric'] == 'P_30') & (result['query'] == 'all')]['value'].values[0])
        top_map.append(result.loc[(result['metric']=='map')&(result['query']=='all')]['value'].values[0])
    return run_id,top_P_10,top_P_20,top_P_30,top_NDCG_10,top_NDCG_20,top_NDCG_30,top_mrr,top_map

# qrels_file = "/home/ubuntu/rupadhyay/2020-derived-qrels/misinfo-qrels-binary.useful-credible"
# runs,top_p_10,top_ndcg_10=get_ndcg_p(qrels_file)


# qrels_file = "/home/ricky/Documents/PhDproject/Project_folder/2020-derived-qrels/misinfo-qrels.useful.wise"
# runs,top_p_10,top_ndcg_10=get_ndcg_p(qrels_file)
qrels_file = "/home/ubuntu/rupadhyay/2020-derived-qrels/misinfo-qrels.cred.wise.test"
# qrels=pd.read_csv(qrels_file,sep=' ',names=['qid','Q0','docno','grade'])
# qid=np.unique(pd.read_csv("./test_qid.csv",sep=';')['qid'].values)
# qrels=qrels[qrels['qid'].isin(qid)]
# qrels.to_csv('/home/ubuntu/rupadhyay/2020-derived-qrels/misinfo-qrels.cred.wise.test',sep=' ',index=None,header=None)
run_id,top_P_10,top_P_20,top_P_30,top_NDCG_10,top_NDCG_20,top_NDCG_30,top_mrr,top_map=get_ndcg_p(qrels_file)

result = pd.DataFrame(
    {'run_id': run_id,
     'ndcg10': top_NDCG_10,
     'ndcg20': top_NDCG_20,
     'ndcg30': top_NDCG_30,
     'p10': top_P_10,
    'p20': top_P_20,
    'p30': top_P_30,
     'mrr':top_mrr,
     'map':top_map
    })

result.to_csv('./result_combined_file_100.csv',sep='\t',index=None)


#best cross_encoder_bert_base_5_8_c_score