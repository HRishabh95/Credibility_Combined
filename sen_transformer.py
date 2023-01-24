import datasets
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from tqdm.auto import tqdm  # so we see progress bar
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers import losses
import math

dataset = datasets.load_dataset("json", data_files={"train": ["train.jsonl"]})
print(f"after: {len(dataset)} rows")

train_samples = []
for row in tqdm(dataset['train']):
    c_score = f'''The credibility score of the document is {str(row['c_score'])}. '''
    #c_score = f'''[SEP] {row['c_score']} '''
    #c_score=''
    train_samples.append(InputExample(
        texts=[row['query'], c_score+row['text']], label=float(row['label'])
    ))

batch_size = 2

dataset = datasets.load_dataset("json", data_files={"test": ["test.jsonl"]})

test_samples = []
for sample in tqdm(dataset['test']):
    c_score = f'''The credibility score of the document is {str(sample['c_score'])}. '''
    #c_score = f'''[SEP] {sample['c_score']} ''' # Acc:0.7, F1- 0.69
    #c_score=''
    test_samples.append(InputExample(
        texts=[sample['query'], c_score+sample['text']],
        label=float(sample['label'])
    ))


evaluator = BinaryClassificationEvaluator.from_input_examples(
    test_samples, write_csv=True
)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

bert = models.Transformer('bert-base-uncased', max_seq_length=510)
pooler = models.Pooling(
    bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

model = SentenceTransformer(modules=[bert, pooler])


loss = losses.OnlineContrastiveLoss(model=model)

epochs = 4

warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1) #10% of train data for warm-up

model.fit(
    train_objectives=[(train_dataloader, loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    evaluator=evaluator,
    evaluation_steps=10,
    output_path='./models/sbert_test_mnr2_c_score',
    show_progress_bar=True,
    optimizer_class=torch.optim.AdamW,
)

print(evaluator(model, output_path='./models/sbert_test_mnr2_c_score/eval/'))
