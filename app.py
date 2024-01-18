#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import streamlit as st
from transformers import BertModel, BertTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


# Load the model

class Bert_Classifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(Bert_Classifier, self).__init__()
        # Specify hidden size of BERT, hidden size of the classifier, and number of labels
        n_input = 768
        n_hidden = 50
        n_output = 5    #Change to 6 if using all labels

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

# Instantiate your model with the appropriate arguments
loaded_model = Bert_Classifier()
loaded_model.load_state_dict(torch.load('bert_classifier.pth'))
loaded_model.eval()

# Instantiate the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[4]:


st.title("Cyber Bullying Detector")
input_text = st.text_input("Enter the message")

#Sentiments
sentiments = ['non','age','ethnicity','gender/body shaming','religion'] #change according to the labels used

if st.button('predict'):
    # Tokenize the input text
    tokens = tokenizer.encode_plus(input_text, add_special_tokens=True,return_token_type_ids=False, return_tensors='pt')
    
    # Perform inference
    with torch.no_grad():
        outputs = loaded_model(**tokens)

    # Predict
    logits = outputs
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Display
    st.header(sentiments[predicted_class])

