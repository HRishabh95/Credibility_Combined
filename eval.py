import torch
import pandas as pd
from datasets import load_metric
from torch.utils.data import DataLoader
from utils import *
from BERT_BCE.CredClassifier import CredClassifier
from BERT_BCE.CredDataset import CredDataset
from BERT_BCE.test import test_prediction

path_to_model = '/home/ricky/PycharmProjects/BERT_TC_train/models/albert-base-v2_lr_2e-05_val_loss_2.7481_ep_4.pt'
# path_to_model = '/content/models/...'  # You can add here your trained model
bert_model = "albert-base-v2"
bs=1
maxlen = 510
path_to_output_file = 'results/output.txt'
df_test=pd.read_csv("test.csv",sep=';')
# Creating instances of training and validation set

print("Reading Test data...")
test_set = CredDataset(df_test, maxlen, bert_model)
test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=CredClassifier(bert_model)
model.load_state_dict(torch.load(path_to_model))
model.to(device)

test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True, result_file=path_to_output_file)


path_to_output_file = 'results/output.txt'  # path to the file with prediction probabilities

labels_test = df_test['label']  # true labels

probs_test = pd.read_csv(path_to_output_file, header=None)[0]  # prediction probabilities

threshold = Find_Optimal_Cutoff(labels_test,probs_test)[0]   # you can adjust this threshold for your own dataset

preds_test=(probs_test>=threshold).astype('uint8') # predicted labels using the above fixed threshold

metric = load_metric("glue", "mrpc")
print(metric._compute(predictions=preds_test, references=labels_test))