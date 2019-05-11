import random 
import numpy as np
import pickle

from keras.preprocessing.text import one_hot, text_to_word_sequence, Tokenizer


### CREATE {id: (tokenized_sentence, isFake)} using keras' Tokenizer, output to file
### CREATE matching embedding matrix, output to file
### COMPUTE average fraction of sentences that have matching words in fasttext (+std)

with open('./data/reviews.pickle', 'rb') as f:
    reviews = pickle.load(f)

with open('./data/valid.pickle', 'rb') as f:
    valid = pickle.load(f)

with open('./data/invalid.pickle', 'rb') as f:
    invalid = pickle.load(f)

