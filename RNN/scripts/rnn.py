from keras.models import Model
from keras.layers import Input, Embedding, LSTM
from keras.layers.wrappers import Bidirectional
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.layers.merge import Add


def singleRNN(embedding_matrix, trainable=False, lr=1e-3):
	rnn_in = Input(shape=(None,), name='rnn_in')
	rnn = Embedding(input_dim=embedding_matrix.shape[0], 
	                output_dim=embedding_matrix.shape[1], 
	                input_length=None,
	                weights=[embedding_matrix],
	                embeddings_initializer=Constant(embedding_matrix),
	                trainable=trainable,
	                mask_zero=True,
	                name="embedding")(rnn_in)

	rnn = Bidirectional(LSTM(256, return_sequences=True, name="lstm_1"), name="bi_lstm_1")(rnn)
	rnn = Bidirectional(LSTM(64, return_sequences=True, name="lstm_2"), name="bi_lstm_2")(rnn)
	rnn_out = LSTM(1, activation='sigmoid', return_sequences=False, name="lstm_out")(rnn)

	rnn_single = Model(inputs=rnn_in,
	            outputs=rnn_out)
	rnn_single.compile(loss='binary_crossentropy',
	            optimizer=Adam(lr=lr),
	            metrics=['accuracy'])
	return rnn_single

def dualRNN(embedding_matrix_inv, embedding_matrix_oov, trainable_inv=False, trainable_oov=True, lr=1e-3):
	rnn_in0 = Input(shape=(None,), name='rnn_in_inv')
	rnn0 = Embedding(input_dim=embedding_matrix_inv.shape[0], 
	                output_dim=embedding_matrix_inv.shape[1], 
	                input_length=None,
	                weights=[embedding_matrix_inv],
	                embeddings_initializer=Constant(embedding_matrix_inv),
	                trainable=trainable_inv,
	                mask_zero=True,
	                name="embedding0")(rnn_in0)

	rnn_in1 = Input(shape=(None,), name='rnn_in_oov')
	rnn1 = Embedding(input_dim=embedding_matrix_oov.shape[0], 
	                output_dim=embedding_matrix_oov.shape[1], 
	                input_length=None,
	                weights=[embedding_matrix_oov],
	                embeddings_initializer=Constant(embedding_matrix_oov),
	                trainable=trainable_oov,
	                mask_zero=True,
	                name="embedding1")(rnn_in1)

	rnn = Add(name="add")([rnn0, rnn1])
	rnn = Bidirectional(LSTM(256, return_sequences=True, name="lstm_1"), name="bi_lstm_1")(rnn)
	rnn = Bidirectional(LSTM(64, return_sequences=True, name="lstm_2"), name="bi_lstm_2")(rnn)
	rnn_out = LSTM(1, activation='sigmoid', return_sequences=False, name="lstm_out")(rnn)

	rnn_dual = Model(inputs=[rnn_in0, rnn_in1],
	            outputs=rnn_out)
	rnn_dual.compile(loss='binary_crossentropy',
	            optimizer=Adam(lr=lr),
	            metrics=['accuracy'])
	return rnn_dual