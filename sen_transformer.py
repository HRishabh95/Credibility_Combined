import datasets
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from tqdm.auto import tqdm  # so we see progress bar
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers import losses




dataset = datasets.load_dataset("json", data_files={"train": ["train.jsonl"]})
print(f"after: {len(dataset)} rows")

train_samples = []
for row in tqdm(dataset['train']):
    train_samples.append(InputExample(
        texts=[row['query'], row['text'], row['c_score']], label=row['label']
    ))

batch_size = 2
loader = DataLoader(
    train_samples, shuffle=True, batch_size=batch_size)

dataset = datasets.load_dataset("json", data_files={"test": ["test.jsonl"]})

samples = []
for sample in tqdm(dataset['test']):
    samples.append(InputExample(
        texts=[sample['query'], sample['text'], sample['c_score']],
        label=sample['label']
    ))



evaluator = BinaryClassificationEvaluator.from_input_examples(
    samples, write_csv=True
)

bert = models.Transformer('bert-base-uncased', max_seq_length=510)
pooler = models.Pooling(
    bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

model = SentenceTransformer(modules=[bert, pooler])


loss = losses.OnlineContrastiveLoss(model=model)

epochs = 50
warmup_steps = int(len(loader) * epochs * 0.1)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    evaluator=evaluator,
    evaluation_steps=50,
    output_path='./models/sbert_test_mnr2',
    show_progress_bar=True,
    optimizer_class=torch.optim.AdamW,
    scheduler='warmupcosinewithhardrestarts',
)

print(evaluator(model, output_path='./models/sbert_test_mnr2/eval/'))
