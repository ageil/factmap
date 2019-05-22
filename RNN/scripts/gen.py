import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, partition, mode="train", all_text=False, train_oov=False, batch_size=16, return_id=False):
        """Initialization"""
        assert type(train_oov) == bool
        assert type(all_text) == bool
        assert mode in partition.keys()
        assert batch_size > 0
        assert type(return_id) == bool
        
        self.partition = partition
        self.mode = mode
        self.all_text = all_text
        self.train_oov = train_oov
        self.batch_size = batch_size
        self.return_id = return_id
        self.IDs = list(partition[mode].keys())
        self.shuffle = True if mode == "train" else False
        self.idx = 0
        self.on_epoch_end() # set self.ID_queue on init


    def __len__(self):
        """Get number of batches per epoch"""
        batches_per_epoch = int(len(self.IDs) // self.batch_size)
        return batches_per_epoch

    def __getitem__(self, idx):
        """Get batch of IDs"""
        batch_IDs = self.ID_queue[idx*self.batch_size : (idx+1)*self.batch_size]
        X, y = self.__load_batch(batch_IDs)
        return X, y

    def on_epoch_end(self):
        """Update ID queue on epoch end"""
        self.ID_queue = []

        if self.shuffle:
            np.random.shuffle(self.IDs)

        # prepare sequence with every ID occurring exactly once
        self.ID_queue += self.IDs
        
    def __get_batch_length(self, batch_IDs):
        """Get max sentence length in batch"""
        if not self.all_text and not self.train_oov:
            max_length = max([len(self.partition[self.mode][ID]['rating']) 
                              for ID in batch_IDs])
        elif self.all_text and not self.train_oov:
            max_length = max([len(self.partition[self.mode][ID]['rating']) +
                              len(self.partition[self.mode][ID]['claim'])
                              for ID in batch_IDs])
        elif not self.all_text and self.train_oov:
            max_inv = max([len(self.partition[self.mode][ID]['rating_inv']) 
                              for ID in batch_IDs])
            max_oov = max([len(self.partition[self.mode][ID]['rating_oov']) 
                              for ID in batch_IDs])
            max_length = max(max_inv, max_oov)
        elif self.all_text and self.train_oov:
            # rating length
            max_rating_inv = max([len(self.partition[self.mode][ID]['rating_inv']) 
                                  for ID in batch_IDs])
            max_rating_oov = max([len(self.partition[self.mode][ID]['rating_oov']) 
                                  for ID in batch_IDs])
            max_rating = max(max_rating_inv, max_rating_oov)
            
            # claim length
            max_claim_inv = max([len(self.partition[self.mode][ID]['claim_inv']) 
                                 for ID in batch_IDs])
            max_claim_oov = max([len(self.partition[self.mode][ID]['claim_oov']) 
                                 for ID in batch_IDs])
            max_claim = max(max_claim_inv, max_rating_oov)
            
            # max length
            max_length = max_rating + max_claim

        return max_length

    def __load_batch(self, batch_IDs):
        """Load batch data"""
        # get batch max sentence length
        max_length = self.__get_batch_length(batch_IDs)
        
        # generate batch
        if self.mode != "predict":
            y = np.zeros(shape=(self.batch_size, 1), dtype=int)
            
        if self.train_oov:
            X_inv = np.zeros(shape=(self.batch_size, max_length), dtype=int)
            X_oov = np.zeros(shape=(self.batch_size, max_length), dtype=int)
            
        else:
            X = np.zeros(shape=(self.batch_size, max_length), dtype=int)

        # fill batch w/ data
        for i, ID in enumerate(batch_IDs):
            if not self.all_text and not self.train_oov:
                rating = self.partition[self.mode][ID]['rating']
                X[i,:len(rating)] = rating
            elif self.all_text and not self.train_oov:
                rating = self.partition[self.mode][ID]['rating']
                claim = self.partition[self.mode][ID]['claim']
                seq = np.concatenate([rating, claim])
                X[i,:len(seq)] = seq
            elif not self.all_text and self.train_oov:
                rating_inv = self.partition[self.mode][ID]['rating_inv']
                rating_oov = self.partition[self.mode][ID]['rating_oov']
                X_inv[i,:len(rating_inv)] = rating_inv
                X_oov[i,:len(rating_oov)] = rating_oov
                X = [X_inv, X_oov]
            elif self.all_text and self.train_oov:
                rating_inv = self.partition[self.mode][ID]['rating_inv']
                rating_oov = self.partition[self.mode][ID]['rating_oov']
                claim_inv = self.partition[self.mode][ID]['claim_inv']
                claim_oov = self.partition[self.mode][ID]['claim_oov']
                seq_inv = np.concatenate([rating_inv, claim_inv]) if rating_inv.sum() != 0 else np.concatenate([claim_inv, rating_inv])
                seq_oov = np.concatenate([rating_oov, claim_oov]) if rating_oov.sum() != 0 else np.concatenate([claim_oov, rating_oov])
                X_inv[i,:len(seq_inv)] = seq_inv
                X_oov[i,:len(seq_oov)] = seq_oov
                X = [X_inv, X_oov]
                
            if self.mode != "predict":
                tgt = self.partition[self.mode][ID]['isFake']
                y[i,:] = tgt
        
        batch = X, y if self.mode != "predict" else X

        if self.return_id:
            batch = batch_IDs, batch
        return batch