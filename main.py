# Simple test preproccessor program to explore the use of TorchText in the implementation
# of a Neural Machine Translation project
# This code is adapted from the following blog: How to use TorchText for neural machine translation
# https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95



print('Started ...')

import spacyplay

spacyplay.testspacy()

exit()

# Load the source English and German language files

# text_en = open(text_en_name, encoding='utf-8').read().split('\n')
text_en_name = 'C:\\Users\\plong\\OneDrive - Deloitte (O365D)\\Deloitte\\usr\\python\\nmt\\datasets\\wmt\'14 english-german\\train-100.en.txt'
file_en = open(text_en_name, 'r', encoding= 'utf-16')
text_en = file_en.read().split('\n')

text_de_name = 'C:\\Users\\plong\\OneDrive - Deloitte (O365D)\\Deloitte\\usr\\python\\nmt\\datasets\\wmt\'14 english-german\\train-100.de.txt'
file_de = open(text_de_name, 'r', encoding= 'utf-16')
text_de = file_de.read().split('\n')

# Tokenize and index the English and German text rows
# Field has been deprecated in torchtext v0.9 - refactor tokenization logic


import spacy

#from torchtext.data import Field, BucketIterator, TabularDataset

# Spacy splits the character stream into token other than just by space, as it understands the root of words
# so "don't" is split into "do" and "n't"

en = spacy.load('en')
de = spacy.load('de')

def tokenize_en(sentence):
    return [token.text for token in en.tokenizer(sentence)]

def tokenize_de(sentence):
    return [token.text for token in de.tokenizer(sentence)]
"""
EN_TEXT = Field(tokenize=tokenize_en)
DE_TEXT = Field(tokenize=tokenize_de, init_token = "<sos>", eos_token = "<eos>")
"""

EN_TEXT = Field(tokenize=tokenize_en)

# Turn langauge datasets into a table with rows for English and corresponding German text, which we split into training
# and test/validation datasets

import pandas as pd

raw_data = {'English': [line for line in text_en], 'German': [line for line in text_de]}

df = pd.DataFrame(raw_data, columns=["English", "German"])

# Remove very long sentences and sentences where transactions are not roughly equal length - not that is completely arbitrary

df['en_len'] = df['English'].str.count(' ')
df['de_len'] = df['German'].str.count(' ')
df = df.query('de_len < 80 & en_len < 80')
df = df.query('de_len < en_len * 1.5 & de_len * 1.5 > en_len')

from sklearn.model_selection import train_test_split
# create train and validation set
train, val = train_test_split(df, test_size=0.1)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)



print(df['English'].iloc[0])

print('... Finished')
