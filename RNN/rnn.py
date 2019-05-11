import numpy as np
import pickle
import argparse
import os

from keras.models import Model
from keras.layers import Input, Embedding, LSTM
from keras.layers.merge import add
from keras.layers.wrappers import Bidirectional
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from scripts.TBCallbacks import TrainValTensorBoard

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="name of the model")
parser.add_argument("--learn_rate", default=1e-3, type=float, help="learning rate")
parser.add_argument("--max_epochs", default=50, type=int, help="maximum number of epochs")
parser.add_argument("--batch_size", default=16, type=int, help="number of samples in each batch, if oversampling use batch_size >= 3")
args = parser.parse_args()


# Set max caption length
max_length = 500   # max sentence length (computed in notebook)
vocab_size = 7607  # total vocab size (computed in notebook)


# load data
with open('./data/data_rated.pickle', 'rb') as f:
    rated_data = pickle.load(f)

with open('./data/embedding_matrix.pickle', 'rb') as f:
    embedding_matrix = pickle.load(f)

with open('./data/embedding_matrix_oov.pickle', 'rb') as f:
    embedding_matrix_oov = pickle.load(f)


# shuffle IDs and split into train/val
ids = list(rated_data.keys())
np.random.seed(2)
np.random.shuffle(ids)

trainFrac = 0.7
validFrac = 0.2

partition = dict()
partition["train"] = {ID: rated_data[ID] for ID in ids[:int(len(ids)*trainFrac)]}
partition["valid"] = {ID: rated_data[ID] for ID in ids[int(len(ids)*trainFrac):int(len(ids)*(trainFrac + validFrac))]}
partition["test"] = {ID: rated_data[ID] for ID in ids[int(len(ids)*(trainFrac + validFrac)):]}


# Define data generator
def generator(partition, mode="train", batch_size=16, predict=False):
    X = np.zeros(shape=(batch_size, max_length), dtype=int)
    X_oov = X = np.zeros(shape=(batch_size, max_length), dtype=int)
    if not predict:
        y = np.zeros(shape=(batch_size, 1), dtype=int)
    val_idx = 0

    while True:
        ids = list(partition[mode].keys())
        if mode == "train":
            batch_ids = np.random.choice(ids, batch_size)
        elif mode == "valid":
            batch_ids = ids[val_idx:val_idx+batch_size]  # eval every sample once
            val_idx += batch_size
        
        for i, ID in enumerate(batch_ids):
            (seq, seq_oov), tgt = partition[mode][ID]
            X[i,:] = seq
            X_oov[i,:] = seq_oov
            if not predict:
                y[i,:] = tgt
            
            out = [X, X_oov], y if not predict else [X, X_oov]
            yield out


# Create train and valid data generators
batch_size = args.batch_size
train_steps_per_epoch = len(partition['train'].keys()) // batch_size
valid_steps_per_epoch = len(partition['valid'].keys()) // batch_size

gen_train = generator(partition=partition, mode='train', batch_size=batch_size)
gen_valid = generator(partition=partition, mode='valid', batch_size=batch_size)


# Build RNN
rnn_in = Input(shape=(max_length,), name='rnn_in')
rnn = Embedding(input_dim=vocab_size, 
                output_dim=300, 
                input_length=max_length,
                weights=[embedding_matrix],
                embeddings_initializer=Constant(embedding_matrix),
                trainable=False,
                mask_zero=True,
                name="embedding")(rnn_in)

rnn_in_oov = Input(shape=(max_length,), name='rnn_in_oov')
rnn_oov = Embedding(input_dim=vocab_size, 
                    output_dim=300, 
                    input_length=max_length,
                    weights=[embedding_matrix_oov],
                    embeddings_initializer=Constant(embedding_matrix_oov),
                    trainable=True,
                    mask_zero=True,
                    name="embedding_oov")(rnn_in_oov)

rnn = add([rnn, rnn_oov], name='add')
rnn = Bidirectional(LSTM(32, return_sequences=True), name="bi_lstm_1")(rnn)
rnn = Bidirectional(LSTM(32, return_sequences=True), name="bi_lstm_2")(rnn)
rnn_out = LSTM(1, activation='sigmoid', return_sequences=False, name="lstm_out")(rnn)
# rnn_out = Dense(1, activation='sigmoid')(rnn)

rnn = Model(inputs=[rnn_in, rnn_in_oov], outputs=rnn_out)

rnn.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=1e-3),
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

hist = rnn.fit_generator(generator=gen_train, 
                         steps_per_epoch=train_steps_per_epoch,
                         epochs=n_epochs,
                         validation_data=gen_valid,
                         validation_steps=valid_steps_per_epoch,
                         verbose=1)


# Save final model
finalsave = savepath + "epoch_{0:03d}".format(args.max_epochs) + "_final.hdf5"
rnn.save(finalsave, include_optimizer=True, overwrite=True)


# Dump history to disk
with open("./output/"+args.name+"/history.pkl", 'wb') as f:
    pickle.dump(hist.history, f, pickle.HIGHEST_PROTOCOL)
