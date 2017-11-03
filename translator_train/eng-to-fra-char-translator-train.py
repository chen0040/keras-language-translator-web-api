from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_UNITS = 256
NUM_SAMPLES = 10000
DATA_PATH = 'data/fra.txt'

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')

for line in lines[: min(NUM_SAMPLES, len(lines)-1)]:
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_char2idx = dict([(char, i) for i, char in enumerate(input_characters)])
input_idx2char = dict([(i, char) for i, char in enumerate(input_characters)])
target_char2idx = dict([(char, i) for i, char in enumerate(target_characters)])
target_idx2char = dict([(i, char) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_char2idx[char]] = 1
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_char2idx[char]] = 1
        if t > 0:
            decoder_target_data[i, t-1, target_char2idx[char]] = 1


encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(units=HIDDEN_UNITS, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)


encoder_model = Model(encoder_inputs, encoder_states)

json = encoder_model.to_json()
open('models/eng-to-fra-char-encoder-architecture.json', 'w').write(json)
encoder_model.save('models/eng-to-fra-char-encoder-weights.h5')

decoder_state_inputs = [Input(shape=(HIDDEN_UNITS, )), Input(shape=(HIDDEN_UNITS, ))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

json = decoder_model.to_json()
open('models/eng-to-fra-char-decoder-architecture.json', 'w').write(json)
decoder_model.save('models/eng-to-fra-char-decoder-weights.h5')





