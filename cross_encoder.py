import datasets
from tqdm.auto import tqdm  # so we see progress bar
from torch.utils.data import DataLoader
import math
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
import torch
import numpy as np

torch.manual_seed(47)
cred_score=True

best_score=0
result_file='/home/ricky/PycharmProjects/BERT_TC_train/cross_encoder_result/result/c_score.csv'

# GRID Training

result=np.zeros((5,10))
for batch in range(1,6):
    for epoch in range(1,11):
        dataset = datasets.load_dataset("json", data_files={"train": ["train.jsonl"]})
        print(f"after: {len(dataset)} rows")

        train_samples = []
        for row in tqdm(dataset['train']):
            if cred_score:
                c_score = f'''The credibility score of the document is {str(row['c_score'])}. '''
            else:
                c_score = ''
            train_samples.append(InputExample(
                texts=[row['query'], c_score+row['text']], label=float(row['label'])
            ))

        batch_size = batch



        dataset = datasets.load_dataset("json", data_files={"test": ["test.jsonl"]})

        test_samples = []
        test_samples_list=[]
        for sample in tqdm(dataset['test']):
            test_samples_list.append([sample['query'],sample['text'],sample['c_score']])
            if cred_score:
                c_score = f'''The credibility score of the document is {str(sample['c_score'])}. '''
            else:
                c_score = ''
            test_samples.append(InputExample(
                texts=[sample['query'], c_score+sample['text']],
                label=float(sample['label'])
            ))

        model = CrossEncoder('distilbert-base-uncased', num_labels=1,max_length=510)
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

        evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name='TREC-dev')

        num_epochs = epoch
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
        if cred_score:
            model_save_path=f'''./cross_encoder_result/cross_encoder_{epoch}_{batch_size}'''+'c_score'
        else:
            model_save_path = f'''./cross_encoder_result/cross_encoder_{epoch}_{batch_size}'''

        model.fit(train_dataloader=train_dataloader,
                  evaluator=evaluator,
                  epochs=num_epochs,
                  evaluation_steps=5000,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path,
                  show_progress_bar=False)
        score = evaluator(model)
        result[batch - 1, epoch - 1] = score
        print("batch: ", batch, "epochs:", epoch, 'acc  {:.3f}'.format(score))
        if score > best_score:
            best_score = score
            best_lr_combination = (batch, epoch)

np.savetxt(result_file,result,delimiter=';')