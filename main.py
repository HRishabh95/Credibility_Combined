from transformers import AdamW, get_linear_schedule_with_warmup

from utils import *
import torch
from CredDataset import *
from CredClassifier import *
from train import train_bert

bert_model = "albert-base-v2"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
freeze_bert = False  # if True, freeze the encoder weights and only update the classification layer weights
maxlen = 510  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
bs = 2  # batch size
iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
lr = 2e-5  # learning rate
epochs = 4  # number of training epochs

set_seed(1)
import pandas as pd

df_train=pd.read_csv("train.csv",sep=';')
df_test=pd.read_csv("test.csv",sep=';')
# Creating instances of training and validation set
print("Reading training data...")
train_set = CredDataset(df_train, maxlen, bert_model)
print("Reading validation data...")
val_set = CredDataset(df_test, maxlen, bert_model)
# Creating instances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size=bs, num_workers=5)
val_loader = DataLoader(val_set, batch_size=bs, num_workers=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = CredClassifier(bert_model, freeze_bert=freeze_bert)

if torch.cuda.device_count() > 1:  # if multiple GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

net.to(device)

criterion = nn.BCEWithLogitsLoss()
opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
num_warmup_steps = 0 # The number of steps for the warmup phase.
num_training_steps = epochs * len(train_loader)  # The total number of training steps
t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate,device)