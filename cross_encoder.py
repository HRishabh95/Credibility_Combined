import datasets
from tqdm.auto import tqdm  # so we see progress bar
from torch.utils.data import DataLoader
import math
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime


dataset = datasets.load_dataset("json", data_files={"train": ["train.jsonl"]})
print(f"after: {len(dataset)} rows")

train_samples = []
for row in tqdm(dataset['train']):
    #c_score = f'''The credibility score of the document is {str(row['c_score'])}. '''
    #c_score = f'''[SEP] {row['c_score']} '''
    c_score=''
    train_samples.append(InputExample(
        texts=[row['query'], c_score+row['text']], label=float(row['label'])
    ))

batch_size = 4



dataset = datasets.load_dataset("json", data_files={"test": ["test.jsonl"]})

test_samples = []
test_samples_list=[]
for sample in tqdm(dataset['test']):
    test_samples_list.append([sample['query'],sample['text'],sample['c_score']])
    #c_score = f'''The credibility score of the document is {str(sample['c_score'])}. '''
    #c_score = f'''[SEP] {sample['c_score']} ''' # Acc:0.7, F1- 0.69
    c_score=''
    test_samples.append(InputExample(
        texts=[sample['query'], c_score+sample['text']],
        label=float(sample['label'])
    ))

model = CrossEncoder('distilbert-base-uncased', num_labels=1,max_length=510)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name='TREC-dev')

num_epochs = 4
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

model_save_path = f'''output/training_trec-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{num_epochs}'''

model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)