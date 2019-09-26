from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, concatenate, Input
from keras_contrib.layers import CRF
from model.lattice_lstm import LatticeLSTM
from model.rnn_layers import MultiInputLSTM, AttentionLSTM

import process_data
import pickle

EMBED_DIM = 100
BiRNN_UNITS = 256
# batch_size=16



def create_model(train=True):
    
   
    if train:
        (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    print(train_x.shape)
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
#     model.add(MultiInputLSTM(BiRNN_UNITS, [50, 50], return_sequences=True))
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)

def create_model2(train=True):
    
    if train:
        (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    
    input1 = Input(shape=(None,), name='input1')
    input2 = Input(shape=(None,), name='input2')
    
    x1 = Embedding(len(vocab), EMBED_DIM, mask_zero=True)(input1)
    x2 = Embedding(len(vocab), EMBED_DIM, mask_zero=True)(input2)
    
    x = concatenate([x1, x2])
    
    x = Bidirectional(MultiInputLSTM(BiRNN_UNITS // 2, input_lengths=[EMBED_DIM, EMBED_DIM], return_sequences=True))(x)
#     x = (MultiInputLSTM(BiRNN_UNITS // 2, input_lengths=[EMBED_DIM, EMBED_DIM], return_sequences=True))(x)
    crf = CRF(len(chunk_tags), sparse_target=True)
    output = crf(x)
    model = Model(inputs=[input1, input2], outputs=output)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
    
    
