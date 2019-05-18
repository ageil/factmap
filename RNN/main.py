import numpy as np
import pickle
import argparse
import os

from keras.models import Model
from keras.layers import Input, Embedding, LSTM
from keras.layers.wrappers import Bidirectional
from keras.initializers import Constant
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from scripts.TBCallbacks import TrainValTensorBoard
from scripts.gen import DataGenerator
from scripts.rnn import singleRNN, dualRNN


parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="name of the model")
parser.add_argument("--all_text", default=False, action='store_true', help="train only based on ratings (True) or also on claim titles (False)")
parser.add_argument("--train_oov", default=False, action='store_true', help="enables dual model with training of out of vocabulary words if true")
parser.add_argument("--train_all", default=False, action='store_true', help="enables training of all embeddings")
parser.add_argument("--dropout", default=False, action='store_true', help="enables 50 pct dropout between LSTM layers")
parser.add_argument("--learn_rate", default=1e-3, type=float, help="learning rate")
parser.add_argument("--max_epochs", default=50, type=int, help="maximum number of epochs")
parser.add_argument("--batch_size", default=16, type=int, help="number of samples in each batch, if oversampling use batch_size >= 3")
args = parser.parse_args()


print("name:", args.name)
print("all_text:", args.all_text)
print("train_oov:", args.train_oov)

# load data
with open('./data/partition.pickle', 'rb') as f:
	partition = pickle.load(f)

if args.train_oov:
	with open('./data/embedding_matrix_inv.pickle', 'rb') as f:
		embedding_matrix_inv = pickle.load(f)
	with open('./data/embedding_matrix_oov.pickle', 'rb') as f:
		embedding_matrix_oov = pickle.load(f)
else:
	with open('./data/embedding_matrix.pickle', 'rb') as f:
		embedding_matrix = pickle.load(f)

# Build RNN
if args.train_oov:
	print("Building dual model...")
	rnn = dualRNN(embedding_matrix_inv=embedding_matrix_inv, 
				  embedding_matrix_oov=embedding_matrix_oov, 
				  trainable_inv=args.train_all, 
				  lr=args.learn_rate)  
else: 
	print("Building single model...")
	rnn = singleRNN(embedding_matrix=embedding_matrix, 
					trainable=args.train_all, 
					lr=args.learn_rate)

# Create train and valid data generators
kwargs = {'partition': partition,
		  'all_text': args.all_text,
		  'train_oov': args.train_oov,
		  'batch_size': args.batch_size}
gen_train = DataGenerator(mode='train', **kwargs)
gen_valid = DataGenerator(mode='valid', **kwargs)

# Add callbacks
directory = "./output/" + args.name
os.makedirs(directory) if not os.path.exists(directory) else None
savepath = directory + "/weights/"
os.makedirs(savepath) if not os.path.exists(savepath) else None
modeldir = savepath + "epoch_{epoch:03d}-valloss_{val_loss:.2f}-valacc_{val_acc:.2f}.hdf5"

callbacks = []
checkpoint = ModelCheckpoint(modeldir, monitor='val_loss', save_weights_only=False, save_best_only=True, mode='min', verbose=1)
tensorboard = TrainValTensorBoard(log_dir=directory+"/logs/", histogram_freq=0, write_graph=True, write_images=True) # custom TB writer object
callbacks.append(checkpoint)  # add checkpoints
callbacks.append(tensorboard) # add tensorboard logging

# Fit model
out = rnn.fit_generator(generator=gen_train, 
                        epochs=args.max_epochs,
                        validation_data=gen_valid,
                        callbacks=callbacks,
                        verbose=1)

# Save final model
finalsave = savepath + "final_epoch_{0:03d}".format(args.max_epochs) + ".hdf5"
rnn.save(finalsave, include_optimizer=True, overwrite=True)

# Dump history to disk
with open(directory + "/history.pickle", 'wb') as f:
    pickle.dump(out.history, f, pickle.HIGHEST_PROTOCOL)