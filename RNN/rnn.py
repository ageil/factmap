import random 
import numpy as np
import pickle
import argparse
import os
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from scripts.TBCallbacks import TrainValTensorBoard

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="name of the model")
parser.add_argument("--learn_rate", default=1e-3, type=float, help="learning rate")
parser.add_argument("--max_epochs", default=50, type=int, help="maximum number of epochs")
parser.add_argument("--batch_size", default=16, type=int, help="number of samples in each batch, if oversampling use batch_size >= 3")
args = parser.parse_args()


# load data
with open('./reviews.pickle', 'rb') as f:
    reviews = pickle.load(f)

with open('./valid.pickle', 'rb') as f:
    valid = pickle.load(f)

with open('./invalid.pickle', 'rb') as f:
    invalid = pickle.load(f)


# Find max caption length
max_length = max([len(x['reviewRating']['alternateName']) for x in reviews])

# Find total vocab size
vocab_size = len({w for r in reviews for w in text_to_word_sequence(r['reviewRating']['alternateName'])})

# Create data dict {id: (tokenized_caption, isFake)}
data = dict()
for r in valid:
    words = one_hot(text=r['reviewRating']['alternateName'], n=vocab_size * 1.5)
    words += [0] * (max_length - len(words))  # padding to vocab_length
    diffBest = r['reviewRating']['bestRating'] - r['reviewRating']['ratingValue'] 
    diffWorst = r['reviewRating']['ratingValue'] - r['reviewRating']['worstRating']
    isFake = int(diffBest >= diffWorst)
    data[r['uid']] = words, isFake

# shuffle IDs and split into train/val
ids = list(data.keys())
random.shuffle(ids)

ids = list(data.keys())
random.shuffle(ids)

trainFrac = 0.8
partition = dict()
partition["train"] = {ID: data[ID] for ID in ids[:int(len(ids)*trainFrac)]}
partition["valid"] = {ID: data[ID] for ID in ids[int(len(ids)*trainFrac):]}


# Define data generator
def generator(partition, mode="train", batch_size=args.batch_size):
    X = np.zeros(shape=(batch_size, max_length), dtype=int)
    y = np.zeros(shape=(batch_size, 1), dtype=int)
    val_idx = 0

    while True:
        ids = list(partition[mode].keys())
        if mode == "train":
            batch_ids = random.sample(ids, batch_size)
        elif mode == "valid":
            batch_ids = ids[val_idx:val_idx+batch_size]
            val_idx += batch_size
        
        for i, ID in enumerate(batch_ids):
            seq, tgt = data[ID]
            X[i,:] = seq
            y[i,:] = tgt
            
            yield X, y


# Create train and valid data generators
batch_size = args.batch_size
train_steps_per_epoch = len(partition['train'].keys()) // batch_size
valid_steps_per_epoch = len(partition['valid'].keys()) // batch_size

gen_train = generator(partition=partition, mode='train', batch_size=batch_size)
gen_valid = generator(partition=partition, mode='valid', batch_size=batch_size)


# Build simple RNN
rnn = Sequential()

rnn.add(Embedding(vocab_size, 64, input_length=max_length))
rnn.add(Bidirectional(LSTM(32, return_sequences=True)))
rnn.add(Bidirectional(LSTM(16, return_sequences=False)))
rnn.add(Dense(1))

rnn.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=args.learn_rate),
            metrics=['accuracy'])


# Add callbacks
directory = "./output/"+args.name+"/"
if not os.path.exists(directory):
    os.makedirs(directory)

callbacks = []
savepath = "./output/weights/"
if not os.path.exists(savepath):
    os.makedirs(savepath)
modeldir = savepath + "epoch_{epoch:03d}-valloss_{val_loss:.2f}-valacc_{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(modeldir, monitor='val_loss', save_weights_only=False, save_best_only=True, mode='min', verbose=1)
callbacks.append(checkpoint)

tensorboard= TrainValTensorBoard(log_dir=directory+"logs/", histogram_freq=0, write_graph=True, write_images=True) # custom TB writer object
callbacks.append(tensorboard) # add tensorboard logging


# Fit model
n_epochs = args.max_epochs

rnn.fit_generator(generator=gen_train, 
                  steps_per_epoch=train_steps_per_epoch,
                  epochs=n_epochs,
                  validation_data=gen_valid,
                  validation_steps=valid_steps_per_epoch,
                  verbose=1,
                  use_multiprocessing=True,
                  workers=4
                  )


# Save final model
finalsave = savepath + "epoch_{0:03d}".format(args.max_epochs) + "_final.hdf5"
model.save(finalsave, include_optimizer=True, overwrite=True)


# Dump history to disk
with open("./output/"+args.name+"/history.pkl", 'wb') as f:
    pickle.dump(hist.history, f, pickle.HIGHEST_PROTOCOL)
