from keras.models import Model
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
MAX_VOCAB_SIZE = 4000
DATA_PATH = 'data/fra.txt'

input_counter = Counter()
target_counter = Counter()

lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
for line in lines[: min(NUM_SAMPLES, len(lines)-1)]:
    input_text, target_text = line.split('\t')
    input_words = [w.lower() for w in nltk.word_tokenize(input_text)]
    target_text = '[START] ' + target_text + ' [END]'
    target_words = [w.lower() for w in nltk.word_tokenize(target_text, language='french')]
    for w in input_words:
        input_counter[w] += 1
    for w in target_words:
        target_counter[w] += 1

input_word2idx = dict()
target_word2idx = dict()
for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
    input_word2idx[word[0]] = idx + 2
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx + 3

input_word2idx['[PAD]'] = 0
input_word2idx['[UNK]'] = 1
target_word2idx['[START]'] = 0
target_word2idx['[END]'] = 1
target_word2idx['[UNK]'] = 2

input_idx2word = {(idx, word) for word, idx in input_word2idx.items()}
target_idx2word = {(idx, word) for word, idx in target_word2idx.items()}

num_encoder_tokens = len(input_idx2word)
num_decoder_tokens = len(target_idx2word)

np.save('models/eng-to-fra/eng-to-fra-word-input-word2idx.npy', input_word2idx)
np.save('models/eng-to-fra/eng-to-fra-word-input-idx2word.npy', input_idx2word)
np.save('models/eng-to-fra/eng-to-fra-word-target-word2idx.npy', target_word2idx)
np.save('models/eng-to-fra/eng-to-fra-word-target-idx2word.npy', target_idx2word)

encoder_input_data = []
decoder_input_data = []
decoder_target_data = []

encoder_max_seq_length = 0
decoder_max_seq_length = 0

lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
for line in lines[: min(NUM_SAMPLES, len(lines)-1)]:
    input_text, target_text = line.split('\t')
    target_text = '[START] ' + target_text + ' [END]'
    input_words = [w.lower() for w in nltk.word_tokenize(input_text)]
    target_words = [w.lower() for w in nltk.word_tokenize(target_text, language='french')]
    encoder_input_wids = []
    decoder_input_wids = []
    decoder_target_wids = []
    for w in input_words:
        w2idx = 1
        if w in input_word2idx:
            w2idx = input_word2idx[w]
        encoder_input_wids.append(w2idx)
    for idx, w in enumerate(target_words):
        w2idx = 2
        if w in target_word2idx:
            w2idx = target_word2idx[w]
        decoder_input_wids.append(w2idx)
        if idx > 0:
            decoder_target_wids.append(w2idx)
    encoder_input_data.append(encoder_input_wids)
    decoder_input_data.append(decoder_input_wids)
    decoder_target_data.append(decoder_target_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(decoder_input_wids), decoder_max_seq_length)

encoder_input_data = pad_sequences(encoder_input_data, encoder_max_seq_length)
decoder_input_data = pad_sequences(decoder_input_data, decoder_max_seq_length)
decoder_target_data = np.array(decoder_target_data)


context = dict()
context['num_encoder_tokens'] = num_encoder_tokens
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

np.save('models/eng-to-fra/eng-to-fra-word-context.npy', context)

encoder_inputs = Input(shape=(None, encoder_max_seq_length), name='encoder_inputs')
encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=HIDDEN_UNITS,
                              input_length=encoder_max_seq_length, name='encoder_embedding')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, decoder_max_seq_length), name='decoder_inputs')
decoder_embedding = Embedding(input_dim=num_decoder_tokens, output_dim=HIDDEN_UNITS,
                              input_length=decoder_max_seq_length, name='decoder_embedding')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_embedding(decoder_inputs),
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
          verbose=1, validation_split=0.2)

json = model.to_json()
open('models/eng-to-fra/eng-to-fra-word-architecture.json', 'w').write(json)
model.save_weights('models/eng-to-fra/eng-to-fra-word-weights.h5')







