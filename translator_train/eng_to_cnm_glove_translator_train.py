from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from keras.callbacks import ModelCheckpoint
import nltk
import numpy as np
import os
import zipfile
import sys
import urllib.request

BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_UNITS = 256
NUM_SAMPLES = 10000
MAX_VOCAB_SIZE = 10000
GLOVE_EMBEDDING_SIZE = 100
DATA_PATH = 'data/cmn.txt'

target_counter = Counter()

GLOVE_MODEL = "very_large_data/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'
WEIGHT_FILE_PATH = 'models/eng-to-cmn/eng-to-cmn-glove-weights.h5'
ARCHITECTURE_FILE_PATH = 'models/eng-to-cmn/eng-to-cmn-glove-architecture.json'

def in_white_list(_word):
    for char in _word:
        if char in WHITELIST:
            return True

    return False


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove():
    if not os.path.exists(GLOVE_MODEL):

        glove_zip = 'very_large_data/glove.6B.zip'

        if not os.path.exists('very_large_data'):
            os.makedirs('very_large_data')

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall('very_large_data')
        zip_ref.close()


def load_glove():
    download_glove()
    _word2em = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em

word2em = load_glove()

lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
for line in lines[: min(NUM_SAMPLES, len(lines)-1)]:
    input_text, target_text = line.split('\t')
    input_words = [w for w in nltk.word_tokenize(input_text.lower())]
    target_text = '\t' + target_text + '\n'
    for char in target_text:
        target_counter[char] += 1

target_word2idx = dict()
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx

target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

num_decoder_tokens = len(target_idx2word)

np.save('models/eng-to-cmn/eng-to-cmn-glove-target-word2idx.npy', target_word2idx)
np.save('models/eng-to-cmn/eng-to-cmn-glove-target-idx2word.npy', target_idx2word)

unknown_emb = np.random.randn(GLOVE_EMBEDDING_SIZE)

np.save('models/eng-to-cmn/eng-to-cmn-glove-unknown-emb.npy', unknown_emb)

encoder_max_seq_length = 0
decoder_max_seq_length = 0

input_texts_word2em = []

lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
for line in lines[: min(NUM_SAMPLES, len(lines)-1)]:
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_words = [w for w in nltk.word_tokenize(input_text.lower())]
    encoder_input_wids = []
    for w in input_words:
        em = unknown_emb
        if w in word2em:
            em = word2em[w]
        encoder_input_wids.append(em)

    input_texts_word2em.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_text), decoder_max_seq_length)

encoder_input_data = pad_sequences(input_texts_word2em, encoder_max_seq_length)

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
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

np.save('models/eng-to-cmn/eng-to-cmn-glove-context.npy', context)

encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
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







