import pandas as pd
import json
import random
random.seed(47)


def balance_dataset(data_path,top_n=14,test=4,json_required=False,triplet=True):
    docs_merged_score= pd.read_csv(data_path,sep=';')
    pos_neg_dataset=[]
    no_qid={}
    qids=docs_merged_score['qid'].unique()
    for qid in qids:
        s = ''
        if qid not in no_qid:
            no_qid[qid]=0
        for ii,rows in docs_merged_score[docs_merged_score['qid']==qid].iterrows():
            if rows['label']==0 and s!='neg':
                pos_neg_dataset.append(rows.tolist())
                s='neg'
                no_qid[qid]+=1
                neg=rows['text']
            elif rows['label']==1 and s!='pos':
                pos_neg_dataset.append(rows.tolist())
                s='pos'
                no_qid[qid]+=1

    pos_neg_dataset=pd.DataFrame(pos_neg_dataset,columns=docs_merged_score.columns)
    pos_neg_dataset['c_score']=pos_neg_dataset['c_score'].astype(str)
    qids=[]
    for r,j in no_qid.items():
        if j>=top_n:
            qids.append(r)
    print(f'''No of Query {len(qids)} with more than {top_n} documents''')


    final_dataset_train=[]
    final_dataset_test=[]
    no_of_train = int(top_n - test)
    print(f'''Training documents per query are {no_of_train}''')
    print(f'''Testing documents per query are {top_n-no_of_train}''')
    for qid in qids:
        final_dataset_train.append(pos_neg_dataset[pos_neg_dataset['qid']==qid].head(no_of_train))
        final_dataset_test.append(pos_neg_dataset[pos_neg_dataset['qid']==qid].tail(top_n-no_of_train))

    final_dataset_train=pd.concat(final_dataset_train).to_csv('./train.csv',sep=';')
    final_dataset_test=pd.concat(final_dataset_test).to_csv('./test.csv',sep=';')

    if json_required:
        final_dataset_train=json.loads(final_dataset_train.to_json(orient='records'))
        final_dataset_test=json.loads(final_dataset_test.to_json(orient='records'))

        with open("train.jsonl", "w") as f:
            for item in final_dataset_train:
                f.write(json.dumps(item) + "\n")

        with open("test.jsonl", "w") as f:
            for item in final_dataset_test:
                f.write(json.dumps(item) + "\n")



def get_unbalanced_dataset(data_path='./datset.csv',test=0.3,json_required=True):
    docs_merged_score = pd.read_csv(data_path, sep='\t',error_bad_lines=False)
    pos_neg_dataset = []
    no_qid = {}
    qids = docs_merged_score['qid'].unique()
    test_set=int(len(qids) * test)
    train_set=len(qids)-test_set
    train_qid=random.sample(list(qids),k=train_set)
    test_qid=[i for i in qids if i not in train_qid]
    final_dataset_train = []
    final_dataset_test = []

    for qid in train_qid:
        final_dataset_train.append(docs_merged_score[docs_merged_score['qid'] == qid])
    for qid in test_qid:
        final_dataset_test.append(docs_merged_score[docs_merged_score['qid'] == qid])


    pd.concat(final_dataset_train).to_csv('./train_qid_clef_bm25.csv', sep=';')
    pd.concat(final_dataset_test).to_csv('./test_qid_clef_bm25.csv', sep=';')

    if json_required:
        final_dataset_train = json.loads(pd.concat(final_dataset_train).to_json(orient='records'))
        final_dataset_test = json.loads(pd.concat(final_dataset_test).to_json(orient='records'))

        with open("train_qid_clef_bm25.jsonl", "w") as f:
            for item in final_dataset_train:
                f.write(json.dumps(item) + "\n")

        with open("test_qid_clef_bm25.jsonl", "w") as f:
            for item in final_dataset_test:
                f.write(json.dumps(item) + "\n")



get_unbalanced_dataset('./dataset_clef_bm25_c_score.csv')

