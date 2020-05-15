import torch
import tensorflow as tf
import pandas as pd
import re
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy.random as npr
import numpy as np
import datetime
import random
import time
from transformers import BertForSequenceClassification, AdamW, BertConfig

def preprocessing(tweet):
    #tweet = tweet2[0]
    tweet = tweet.lower() # convert text to lower-case
    tweet.replace('\n', ' ')#Remove non_ASCII
    tweet = re.sub("\s\s+", " ", tweet) #Remove extra spaces
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet) #remove URLs
    tweet = re.sub('rt @[^\s]+', '', tweet) # remove retweet tags 
    tweet = re.sub('@[^\s]+', '', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', '', tweet) # remove the # in #hashtag
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)#Remove non_ASCII
    tweet = re.sub(r"[^0-9a-z ]", "", tweet)#Just remove all non(number, alphabets, spaces)
    return tweet.strip()

def prepare_inputs(samples, labels):
    ids = []
    masks = []

    # For every sentence...
    for t in samples:
      encoded_dict = tkn.encode_plus(t, add_special_tokens = True, max_length = 96, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')
      ids.append(encoded_dict['input_ids'])
      masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(ids, dim=0)
    attention_masks = torch.cat(masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

def get_conf_mat(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    mat = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            mat[i, j] = np.sum((pred_flat == i) & (labels_flat == j))

    return mat


def calc_metric(mat):
    TP = mat[1, 1]
    FP = mat[1, 0]
    FN = mat[0, 1]
    TN = mat[0, 0]
    
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    f1 = 2 * ((precision * recall) / (precision + recall))
    acc = (TP + TN) / float(np.sum(mat))
    return acc, precision, recall, f1

tkn = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

path = 'data1.csv'
data = pd.read_csv(path, delimiter=',',  header=None, names=['colA', 'tweets','labels', 'colD', 'colE', 'colF'])
data = data[['tweets', 'labels']]
data['tweets'] = data['tweets'].apply(preprocessing)
mask = npr.rand(len(data)) < 0.8
train_data = data[mask]
test_data = data[~mask]
train_tweets = train_data.tweets.values
train_labels = train_data.labels.values
test_tweets = test_data.tweets.values
test_labels = test_data.labels.values

train_input_ids, train_attention_masks, train_labels = prepare_inputs(train_tweets, train_labels)
test_input_ids, test_attention_masks, test_labels = prepare_inputs(test_tweets, test_labels)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

batch_size = 32
train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

model = model.cuda()

optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)


epochs = 4

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

training_stats = []
device = 'cuda'

# For each epoch...
for epoch_i in range(0, epochs):
    print("Starting Epoch: %d"%(epoch_i))
    total_train_loss = 0
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()        
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step%40 == 0:
          print("Training loss: {0:.2f}".format(loss))

#Testing the model
model.eval()
test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)
total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0
conf_mat = np.zeros((2,2))

for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    with torch.no_grad():        
        (loss, logits) = model(b_input_ids, 
                               token_type_ids=None, 
                               attention_mask=b_input_mask,
                               labels=b_labels)
        
    total_eval_loss += loss.item()

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    conf_mat += get_conf_mat(logits, label_ids)
    
acc, prec, recall, f1 = calc_metric(conf_mat[::-1, ::-1])
print("Accuracy: %f, Precision: %f, Recall: %f, F1: %f"%(acc, prec, recall, f1))

