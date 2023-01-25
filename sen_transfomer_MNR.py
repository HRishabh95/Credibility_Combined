import datasets
from sentence_transformers import datasets as stdataset
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sentence_transformers import InputExample
from tqdm.auto import tqdm  # so we see progress bar
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers import losses
import math
import numpy as np

torch.manual_seed(47)
cred_score=False

best_score=0
result_file='/home/ricky/PycharmProjects/BERT_TC_train/sent_MNR_models/result/non_c_score.csv'
result=np.zeros((5,10))
for batch in range(1,6):
    for epoch in range(1,11):
        dataset = datasets.load_dataset("json", data_files={"train": ["train.jsonl"]})
        dataset = dataset.filter(
            lambda x: True if x['label'] == 0 else False
        )
        print(f"after: {len(dataset)} rows")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        train_samples = []
        for row in tqdm(dataset['train']):
            if cred_score:
                c_score = f'''The credibility score of the document is {str(row['c_score'])}. '''
            else:
                c_score = ''
            train_samples.append(InputExample(
                texts=[row['query'], c_score+row['text']]
            ))


        batch_size = batch

        loader = stdataset.NoDuplicatesDataLoader(
            train_samples, batch_size=batch_size)

        from sentence_transformers import models, SentenceTransformer

        bert = models.Transformer('bert-base-uncased')
        pooler = models.Pooling(
            bert.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )

        model = SentenceTransformer(modules=[bert, pooler])


        from sentence_transformers import losses

        loss = losses.MultipleNegativesRankingLoss(model)

        epochs = epoch
        warmup_steps = int(len(loader) * epochs * 0.1)

        if cred_score:
            output_path=f'''./sent_MNR_models/sbert_test_{loss._get_name()}_{epochs}_{batch_size}'''+'c_score'
        else:
            output_path = f'''./sent_MNR_models/sbert_test_{loss._get_name()}_{epochs}_{batch_size}'''
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=False
        )

        dataset = datasets.load_dataset("json", data_files={"test": ["test.jsonl"]})


        test_samples = []
        for sample in tqdm(dataset['test']):
            if cred_score:
                c_score = f'''The credibility score of the document is {str(sample['c_score'])}. '''
            else:
                c_score=''
            test_samples.append(InputExample(
                texts=[sample['query'], c_score+sample['text']],
                label=float(sample['label'])
            ))


        evaluator = BinaryClassificationEvaluator.from_input_examples(
            test_samples, write_csv=True
        )


        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(output_path)
        score=evaluator(model,output_path=f'''{output_path}/eval/''')
        result[batch-1,epochs-1]=score
        print("batch: ", batch, "epochs:", epochs, 'acc  {:.3f}'.format(score))
        if score > best_score:
            best_score = score
            best_lr_combination = (batch, epochs)

np.savetxt(result_file,result,delimiter=';')