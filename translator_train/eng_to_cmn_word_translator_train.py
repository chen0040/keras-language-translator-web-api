from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import nltk
import numpy as np

BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_UNITS = 256
NUM_SAMPLES = 10000
MAX_VOCAB_SIZE = 10000
DATA_PATH = 'data/cmn.txt'
WEIGHT_FILE_PATH = 'models/eng-to-cmn/eng-to-cmn-word-weights.h5'
ARCHITECTURE_FILE_PATH = 'models/eng-to-cmn/eng-to-cmn-word-architecture.json'

input_counter = Counter()
target_counter = Counter()

lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
for line in lines[: min(NUM_SAMPLES, len(lines)-1)]:
    input_text, target_text = line.split('\t')
    input_words = [w for w in nltk.word_tokenize(input_text.lower())]
    target_text = '\t' + target_text + '\n'
    for w in input_words:
        input_counter[w] += 1
    for char in target_text:
        target_counter[char] += 1

input_word2idx = dict()
target_word2idx = dict()
for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
    input_word2idx[word[0]] = idx + 2
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx

input_word2idx['PAD'] = 0
input_word2idx['UNK'] = 1

input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])
target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

num_encoder_tokens = len(input_idx2word)
num_decoder_tokens = len(target_idx2word)

np.save('models/eng-to-cmn/eng-to-cmn-word-input-word2idx.npy', input_word2idx)
np.save('models/eng-to-cmn/eng-to-cmn-word-input-idx2word.npy', input_idx2word)
np.save('models/eng-to-cmn/eng-to-cmn-word-target-word2idx.npy', target_word2idx)
np.save('models/eng-to-cmn/eng-to-cmn-word-target-idx2word.npy', target_idx2word)

encoder_input_data = []

encoder_max_seq_length = 0
decoder_max_seq_length = 0

lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
for line in lines[: min(NUM_SAMPLES, len(lines)-1)]:
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_words = [w for w in nltk.word_tokenize(input_text.lower())]
    encoder_input_wids = []
    for w in input_words:
        w2idx = 1  # default [UNK]
        if w in input_word2idx:
            w2idx = input_word2idx[w]
        encoder_input_wids.append(w2idx)

    encoder_input_data.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_text), decoder_max_seq_length)

encoder_input_data = pad_sequences(encoder_input_data, encoder_max_seq_length)

decoder_target_data = np.zeros(shape=(NUM_SAMPLES, decoder_max_seq_length, num_decoder_tokens))
decoder_input_data = np.zeros(shape=(NUM_SAMPLES, decoder_max_seq_length, num_decoder_tokens))
lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
for lineIdx, line in enumerate(lines[: min(NUM_SAMPLES, len(lines)-1)]):
    _, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    for idx, char in enumerate(target_text):
        if char in target_word2idx:
            w2idx = target_word2idx[char]
            decoder_input_data[lineIdx, idx, w2idx] = 1
            if idx > 0:
                decoder_target_data[lineIdx, idx-1, w2idx] = 1

context = dict()
context['num_encoder_tokens'] = num_encoder_tokens
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

np.save('models/eng-to-cmn/eng-to-cmn-word-context.npy', context)

encoder_inputs = Input(shape=(None, ), name='encoder_inputs')
encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=HIDDEN_UNITS,
                              input_length=encoder_max_seq_length, name='encoder_embedding')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


json = model.to_json()
open(ARCHITECTURE_FILE_PATH, 'w').write(json)

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
          verbose=1, validation_split=0.2, callbacks=[checkpoint])

model.save_weights(WEIGHT_FILE_PATH)







