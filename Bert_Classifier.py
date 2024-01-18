#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Libraries for general purpose
import pandas as pd
import numpy as np

# Data preprocessing
import regex as re
from sklearn.model_selection import train_test_split

# PyTorch LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Transformers library for BERT
import transformers
from transformers import BertModel
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import time


# In[6]:


df = pd.read_csv("Fine-Grained and Balanced Cyberbullying Detection Dataset.csv", header=None)
df.head(10)


# In[8]:


df = df.rename(columns={0: 'text', 1: 'sentiment'})
df.head(10)


# In[9]:


sentiments = ['non-cyberbully','age','ethnicity','gender','religion']
#Removing the "others" category of data
df = df[df["sentiment"]!=4]
df['sentiment'] = df['sentiment'].replace(5, 4)
df.reset_index(drop=True, inplace=True)
df.info


# In[10]:


# Light data cleaning as heavy cleaning can remove context

def remove_usernames(txt):
    # Define the regex pattern to match usernames
    pattern = re.compile(r'@\w+')

    # Use sub() to replace matched usernames with an empty string
    cleaned = re.sub(pattern, '', txt)

    return cleaned

def text_clean(x):

    x = remove_usernames(x)
    x = x.lower() # lowercase everything
    x = x.encode('ascii', 'ignore').decode()  # remove unicode characters
    x = re.sub(r'https*\S+', ' ', x) # remove links
    x = re.sub(r'http*\S+', ' ', x)
    # cleaning up text
    x = re.sub(r'\'\w+', '', x) 
    x = re.sub(r'\w*\d+\w*', '', x)
    x = re.sub(r'\s{2,}', ' ', x)
    x = re.sub(r'\s[^\w\s]\s', '', x) 
    x = re.sub(r"\s\s+", " ", x)

    return x


# In[11]:


df['text'] = df['text'].apply(text_clean)


# In[12]:


df.drop_duplicates(subset='text', inplace=True)
df.dropna(subset='text', inplace=True)


# In[13]:


df.isna().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


X = df['text'].values
y = df['sentiment'].values


# In[16]:


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True)


# In[17]:


X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, shuffle=True)


# In[18]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# In[19]:


#Defining constants
MAX_LEN = 128
batch_size = 32
EPOCHS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[20]:


def bert_tokenizer(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]` special tokens
            max_length=MAX_LEN,             # Choose max length to truncate/pad
            padding='max_length',         # Pad sentence to max length 
            return_attention_mask=True      # Return attention mask
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    
    return input_ids, attention_masks


# In[21]:


train_inputs, train_masks = bert_tokenizer(X_train)
val_inputs, val_masks = bert_tokenizer(X_valid)
test_inputs, test_masks = bert_tokenizer(X_test)


# In[22]:


# Convert target columns to pytorch tensors format
train_labels = torch.from_numpy(y_train)
val_labels = torch.from_numpy(y_valid)
test_labels = torch.from_numpy(y_test)


# In[23]:


# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Create the DataLoader for our test set
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


# In[24]:


class Bert_Classifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(Bert_Classifier, self).__init__()
        # Specify hidden size of BERT, hidden size of the classifier, and number of labels
        n_input = 768
        n_hidden = 50
        n_output = 5     #Change to 6 if using all labels

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate the classifier (a fully connected layer followed by a SiLU activation and another fully connected layer)
        self.classifier = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_output)
        )

        # Freeze the BERT model weights if freeze_bert is True (useful for feature extraction without fine-tuning)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Feed input data (input_ids and attention_mask) to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the `[CLS]` token from the BERT output (useful for classification tasks)
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed the extracted hidden state to the classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


# In[25]:


# Function for initializing the BERT Classifier model, optimizer, and learning rate scheduler
def initialize_model(epochs=4):
    # Instantiate Bert Classifier
    bert_classifier = Bert_Classifier(freeze_bert=False)

    bert_classifier.to(device)

    # Set up optimizer
    optimizer = optim.AdamW(bert_classifier.parameters(),
                        lr=5e-5,   #Default value
                        eps=1e-8,
                      )

    # Calculate total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Define the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


# In[26]:


bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS)


# In[27]:


# Define Cross entropy Loss function for the multiclass classification task
loss_fn = nn.CrossEntropyLoss()

def bert_train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):

    print("Start training...\n")
    for epoch_i in range(epochs):
        print("-"*10)
        print("Epoch : {}".format(epoch_i+1))
        print("-"*10)
        print("-"*38)
        print(f"{'BATCH NO.':^7} | {'TRAIN LOSS':^12} | {'ELAPSED (s)':^9}")
        print("-"*38)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        
        ###TRAINING###

        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass and get logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update model parameters:
            # fine tune BERT params and train additional dense layers
            optimizer.step()
            # update learning rate
            scheduler.step()

            # Print the loss values and time elapsed for every 100 batches
            if (step % 100 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                
                print(f"{step:^9} | {batch_loss / batch_counts:^12.6f} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        ###EVALUATION###
        
        # Put the model into the evaluation mode
        model.eval()
        
        # Define empty lists to host accuracy and validation for each batch
        val_accuracy = []
        val_loss = []

        for batch in val_dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = tuple(t.to(device) for t in batch)
            
            # We do not want to update the params during the evaluation,
            # So we specify that we dont want to compute the gradients of the tensors
            # by calling the torch.no_grad() method
            with torch.no_grad():
                logits = model(batch_input_ids, batch_attention_mask)

            loss = loss_fn(logits, batch_labels)

            val_loss.append(loss.item())

            # Get the predictions starting from the logits (get index of highest logit)
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the validation accuracy 
            accuracy = (preds == batch_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)
        
        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch
        print("-"*61)
        print(f"{'AVG TRAIN LOSS':^12} | {'VAL LOSS':^10} | {'VAL ACCURACY (%)':^9} | {'ELAPSED (s)':^9}")
        print("-"*61)
        print(f"{avg_train_loss:^14.6f} | {val_loss:^10.6f} | {val_accuracy:^17.2f} | {time_elapsed:^9.2f}")
        print("-"*61)
        print("\n")
    
    print("Training complete!")


# In[23]:


bert_train(bert_classifier, train_dataloader, val_dataloader, epochs=EPOCHS)


# In[28]:


def bert_predict(model, test_dataloader):
    
    # Define empty list to host the predictions
    preds_list = []
    
    # Put the model into evaluation mode
    model.eval()
    
    for batch in test_dataloader:
        batch_input_ids, batch_attention_mask = tuple(t.to(device) for t in batch)[:2]
        
        # Avoid gradient calculation of tensors by using "no_grad()" method
        with torch.no_grad():
            logit = model(batch_input_ids, batch_attention_mask)
        
        # Get index of highest logit
        pred = torch.argmax(logit,dim=1).cpu().numpy()
        # Append predicted class to list
        preds_list.extend(pred)

    return preds_list


# In[29]:


bert_preds = bert_predict(bert_classifier, test_dataloader)


# In[30]:


print('Classification Report for BERT :\n', classification_report(y_test, bert_preds, target_names=sentiments))

SiLU, 5e-5, 1 layer
94.12,94.42
SiLU, 5e-5, 2 layers
94.10,94.32
SiLU, ReLU, 5e-5 ,2 layers
94.14,94.61
SiLU, 6e-5, 2 layers
94.68, 95.10
SiLU, 6e-5, 1 layer
94.73, 95.29
SiLU, 7e-5, 1 layer
94.75, 94.86
SiLU, 6.5e-5, 1 layer
94.41, 95.08
SiLU, 6.2e-5, 1 layer
94.56, 95.13
SiLU, 5.9e-5, 1 layer
94.37, 94.80
SiLU, 6.05e-5, 1 layer
94.59, 95.07
SiLU, 1e-5, 1 layer
93.48, 94.34
# In[28]:


# Save the model
torch.save(bert_classifier.state_dict(), 'bert_classifier.pth')

