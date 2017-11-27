from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np
import os
import sys
import urllib.request
import zipfile

HIDDEN_UNITS = 256
GLOVE_EMBEDDING_SIZE = 100

VERY_LARGE_DATA_DIR_PATH = '../translator_train/very_large_data'
MODEL_DIR_PATH = '../translator_train/models/eng-to-cmn'
GLOVE_MODEL = VERY_LARGE_DATA_DIR_PATH + "/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'


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

        glove_zip = VERY_LARGE_DATA_DIR_PATH + '/glove.6B.zip'

        if not os.path.exists(VERY_LARGE_DATA_DIR_PATH):
            os.makedirs(VERY_LARGE_DATA_DIR_PATH)

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall(VERY_LARGE_DATA_DIR_PATH)
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


class EngToCmnGloveTranslator(object):
    model = None
    encoder_model = None
    decoder_model = None
    target_word2idx = None
    target_idx2word = None
    max_encoder_seq_length = None
    max_decoder_seq_length = None
    num_decoder_tokens = None
    word2em = None
    unknown_emb = None

    def __init__(self):
        self.word2em = load_glove()
        self.unknown_emb = np.load(MODEL_DIR_PATH + '/eng-to-cmn-glove-unknown-emb.npy')
        self.target_word2idx = np.load(
            MODEL_DIR_PATH + '/eng-to-cmn-glove-target-word2idx.npy').item()
        self.target_idx2word = np.load(
            MODEL_DIR_PATH + '/eng-to-cmn-glove-target-idx2word.npy').item()
        context = np.load(MODEL_DIR_PATH + '/eng-to-cmn-glove-context.npy').item()
        self.max_decoder_seq_length = context['decoder_max_seq_length']
        self.max_encoder_seq_length = context['encoder_max_seq_length']
        self.num_decoder_tokens = context['num_decoder_tokens']

        encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name="encoder_lstm")
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.model.load_weights(MODEL_DIR_PATH + '/eng-to-cmn-glove-weights.h5')
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def translate_lang(self, input_text):
        input_seq = []
        input_wids = []
        for word in nltk.word_tokenize(input_text.lower()):
            emb = self.unknown_emb
            if word in self.word2em:
                emb = self.word2em[word]
            input_wids.append(emb)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_encoder_seq_length)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_word2idx['\t']] = 1
        target_text = ''
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text += sample_word

            if sample_word == '\n' or len(target_text) >= self.max_decoder_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()

    def test_run(self):
        print(self.translate_lang('Be nice.'))
        print(self.translate_lang('Drop it!'))
        print(self.translate_lang('Get out!'))


def main():
    model = EngToCmnGloveTranslator()
    model.test_run()


if __name__ == '__main__':
    main()
