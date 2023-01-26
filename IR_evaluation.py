import datasets
from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models

cred_score=False



dataset = datasets.load_dataset("json", data_files={"test": ["test.jsonl"]})
dataset_pos = dataset.filter(
    lambda x: True if x['label'] == 0 else False
)
print(f"after: {len(dataset)} rows")

dataset_neg = dataset.filter(
    lambda x: True if x['label'] == 1 else False
)

dev_queries={}
dev_rel_docs = {}       #Mapping qid => set with relevant pids
needed_pids=set()
needed_qids=set()
corpus={}
dev_sample={}
for i in dataset_pos['test']:
    if i['qid'] not in dev_sample:
        dev_sample[i['qid']]={
            'query': i['query'],
            'positive': set(),
            'negative':set()
        }
    dev_queries[i['qid']]=i['query']
    if i['qid'] not in dev_rel_docs:
        dev_rel_docs[i['qid']]=set()
    needed_qids.add(i['qid'])
    needed_pids.add(i['docno'])
    dev_rel_docs[i['qid']].add(i['docno'])
    if cred_score:
        c_score = f'''The credibility score of the document is {str(i['c_score'])}. '''
        # c_score = f'''[SEP] {i['c_score']} ''' # Acc:0.7, F1- 0.69

    else:
        c_score = ''
    corpus[i['docno']]=c_score+i['text']

for i in dataset_neg['test']:
    if cred_score:
        c_score = f'''The credibility score of the document is {str(i['c_score'])}. '''
        # c_score = f'''[SEP] {i['c_score']} ''' # Acc:0.7, F1- 0.69

    else:
        c_score = ''
    corpus[i['docno']] = c_score + i['text']

eval=evaluation.InformationRetrievalEvaluator(dev_queries,corpus,dev_rel_docs,show_progress_bar=True,
                                                        corpus_chunk_size=100,
                                                        precision_recall_at_k=[5,10],
                                                        ndcg_at_k=[5,10],
                                                        map_at_k=[5,10],
                                                        mrr_at_k=[5,10],
                                                        name="trec")

model_file='/home/ricky/PycharmProjects/BERT_TC_train/sent_MNR_models/sbert_test_MultipleNegativesRankingLoss_10_3'
model = SentenceTransformer(model_file)
f=eval(model,output_path=f'''{model_file}/''')
