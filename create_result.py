import pandas as pd
import numpy as np

def get_averaged_score(simi_score):
    qids = np.unique(simi_score.qid.values)
    #weights_10 = get_final_cred_score_weights(simi_score)
    weights_10 = [0.25, 0.15, 0.125, 0.125, 0.1, 0.05, 0.05, 0.05,0.05,0.05]
    average_dfs = []
    for qid in qids:
        qid_dfs = simi_score.loc[simi_score['qid'] == qid]
        qid_docs = np.unique(simi_score.loc[simi_score['qid'] == qid].docno.values)
        for qid_doc in qid_docs:
            #weights_20 = [0.12,0.1,0.06,0.06,0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.04,0.04,0.04,0.02,0.02,0.02,0.02,0.01,0.01]

            score = qid_dfs.loc[qid_dfs['docno'] == qid_doc].scores*weights_10
            score_av = np.sum(score)
            average_dfs.append([qid, "Q0", qid_doc, score_av])
    ave_cred = pd.DataFrame(average_dfs, columns=['qid', 'Q0', 'docno', 'c_score'])
    return ave_cred

def get_weights():
    weights=[0.4,0.6]
    return weights
