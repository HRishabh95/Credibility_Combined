from trectools import TrecQrel, procedures
from scipy.stats import hmean


def get_ndcg_p(qrels_file):
    qrels = TrecQrel(qrels_file)
    runs=procedures.list_of_runs_from_path('/home/ricky/PycharmProjects/BERT_TC_train/results','*.csv')
    results = procedures.evaluate_runs(runs, qrels, per_query=True)
    top_NDCG_10,top_P_10=[],[]
    for result in results:
        result=result.data
        top_NDCG_10.append(result.loc[result['metric']=='NDCG_100']['value'].dropna().values.mean())
        top_P_10.append(result.loc[result['metric']=='map']['value'].dropna().values.mean())
    return runs,top_P_10,top_NDCG_10

qrels_file = "/home/ricky/Documents/PhDproject/Project_folder/2020-derived-qrels/misinfo-qrels-binary.useful-credible"
runs,top_p_10,top_ndcg_10=get_ndcg_p(qrels_file)


qrels_file = "/home/ricky/Documents/PhDproject/Project_folder/2020-derived-qrels/misinfo-qrels.useful.wise"
runs,top_p_10,top_ndcg_10=get_ndcg_p(qrels_file)
qrels_file = "/home/ricky/Documents/PhDproject/Project_folder/2020-derived-qrels/misinfo-qrels.cred.wise"
runs,cred_p_10,cred_ndcg_10=get_ndcg_p(qrels_file)

for i in range(len(top_p_10)):
    MM_NDCG=hmean([top_ndcg_10[i],cred_ndcg_10[i]])
    MM_P_10=hmean([top_p_10[i],cred_p_10[i]])
    print(runs[i],MM_NDCG,MM_P_10)