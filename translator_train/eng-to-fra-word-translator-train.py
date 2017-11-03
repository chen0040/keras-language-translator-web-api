from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from collections import Counter
import nltk

BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_UNITS = 256
NUM_SAMPLES = 10000
MAX_VOCAB_SIZE = 2000
DATA_PATH = 'data/fra.txt'

input_counter = Counter()
target_counter = Counter()

lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
for line in lines:
    input_text, target_text = line.split('\t')
    input_words = [w.lower() for w in nltk.word_tokenize(input_text)]
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
    target_word2idx[word[0]] = idx + 2


