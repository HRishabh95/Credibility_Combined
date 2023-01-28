import math
import os.path
from utils import mkdir_p
import datasets
import numpy as np
import torch
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers.evaluation import InformationRetrievalEvaluator

cred_score=True

dataset = datasets.load_dataset("json", data_files={"train": ["train_qid.jsonl"]})

train_samples = []
for row in tqdm(dataset['train']):
    if cred_score:
        c_score = f'''The credibility score of the document is {str(row['c_score'])}. '''
        #c_score = f'''[SEP] {row['c_score']} ''' # Acc:0.7, F1- 0.69
    else:
        c_score = ''
    train_samples.append(InputExample(
        texts=[row['query'], c_score+row['text']], label=float(row['label'])
    ))

dataset = datasets.load_dataset("json", data_files={"test": ["test_qid.jsonl"]})
dataset_pos = dataset.filter(
    lambda x: True if x['label'] == 0 else False
)

dataset_neg = dataset.filter(
    lambda x: True if x['label'] == 1 else False
)

dev_sample={}
for i in dataset_pos['test']:
    if i['qid'] not in dev_sample:
        dev_sample[i['qid']]={
            'query': i['query'],
            'positive': set(),
            'negative':set()
        }
    if i['qid'] in dev_sample:
        if cred_score:
            c_score = f'''The credibility score of the document is {str(i['c_score'])}. '''
            #c_score = f'''[SEP] {i['c_score']} ''' # Acc:0.7, F1- 0.69

        else:
            c_score = ''
        dev_sample[i['qid']]['positive'].add(c_score+i['text'])
    for j in dataset_neg['test']:
        if j['qid']==i['qid']:
            if cred_score:
                c_score = f'''The credibility score of the document is {str(j['c_score'])}. '''
                #c_score = f'''[SEP] {j['c_score']} ''' # Acc:0.7, F1- 0.69
            else:
                c_score = ''
            dev_sample[i['qid']]['negative'].add(c_score+j['text'])


torch.manual_seed(47)
model_name = 'bert-large-uncased'

best_score=0
model_path = f'''./cross_encoder_CRRerank_{model_name}'''
result_folder = f'''{model_path}/result'''
mkdir_p(result_folder)
if cred_score:
    result_file = f'''{model_path}/result/c_score.csv'''
else:
    result_file = f'''{model_path}/result/no_c_score.csv'''

result=np.zeros((7,5))
for batch_number,batch in enumerate([2,4,6,8,10]):
    for epoch_number, epoch in enumerate([6,7,8,9]):
        print(batch,epoch)
        train_batch_size=batch

        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        evaluator = CERerankingEvaluator(dev_sample, name='train-eval')
        if cred_score:
            model_save_path=f'''./cross_encoder_CRRerank_{model_name}/cross_encoder_{epoch}_{train_batch_size}_'''+'c_score'
            mkdir_p(model_save_path)
        else:
            model_save_path = f'''./cross_encoder_CRRerank_{model_name}/cross_encoder_{epoch}_{train_batch_size}'''
            mkdir_p(model_save_path)
            
        if not os.path.isfile(model_save_path+"/pytorch_model.bin"):
            print("Training")
            warmup_steps = math.ceil(len(train_dataloader) * epoch * 0.1)  # 10% of train data for warm-up
            model_name = 'bert-large-uncased'
    
            model = CrossEncoder(model_name, num_labels=1, max_length=510)
    
            # Train the model
            model.fit(train_dataloader=train_dataloader,
                      evaluator=evaluator,
                      epochs=epoch,
                      evaluation_steps=2000,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path,
                      use_amp=True,
                      )
#             score = evaluator(model)
#             result[batch_number, epoch_number] = score
#             print("batch: ", batch, "epochs:", epoch, 'acc  {:.3f}'.format(score))
#             if score > best_score:
#                 best_score = score
#                 best_lr_combination = (batch, epoch)
#
# np.savetxt(result_file,result,delimiter=';')