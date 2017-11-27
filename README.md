# keras-language-translator-web-api

A simple language translator implemented in Keras with Flask serving web

The language translator is built based on seq2seq models, and can infer based on either character-level or word-level. 

The seq2seq model is implemented using LSTM encoder-decoder on Keras. 

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

The translator models are trained using eng-to-french and eng-to-chinese data set and are available in the 
translator_train/models directory. During runtime, the flask app will load these trained models to perform the 
translation.

Currently only the eng-to-chinese and eng-to-french translations models are provided as examples, you can
go to [http://www.manythings.org/anki/](http://www.manythings.org/anki/) to download more datasets for the translator
training and use the scripts in the translator_train to generate new seq2seq for other language translation

## Training (Optional)

As the trained models are already included in the "translator_train/models" folder in the project, the training is
not required. However, if you like to tune the parameters of the seq2seq and retrain the models, you can use the 
following command to run the training:

```bash
cd translator_train
python eng_to_cmn_char_seq2seq_train.py
```

The above commands will train seq2seq model using eng-to-chinese dataset on the character-level and store the trained model
in "translator_train/models/eng-to-cmn/eng-to-cmn-char-**"

If you like to train other models, you can use the same command above on another train python scripts:

* eng_to_cmn_word_translator_train.py: train on eng-to-chinese on word-level (one hot encoding)
* eng_to_cmn_glove_translator_train.py: train on eng-to-chinese on word-level (GloVe encoding)
* eng_to_fra_char_translator_train.py: train on eng-to-french on character-level
* eng_to_fra_word_translator_train.py: train on eng-to-french on word-level (one hot encoding)
* eng_to_fra_glove_translator_train.py: train on eng-to-french on word-level (GloVe encoding)

## Running Web Api Server

Goto translator_web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained seq2seq models:

* Character-level seq2seq models
* Word-level seq2seq models (one hot encoding)
* Word-level seq2seq models (GloVe encoding)

## Invoke Web Api

To translate an english sentence to other languages using web api, after the flask server is started, run the following curl POST query
in your terminal:

```bash
curl -H 'Content-Type application/json' -X POST -d '{"level":"level_type", "sentence":"your_sentence_here", "target_lang":"target_language"}' http://localhost:5000/translate_eng
```

The level_type can be "char" or "word", the target_lang can be "chinese" or "french"

(Note that same results can be obtained by running a curl GET query to http://localhost:5000/translate_eng?sentence=your_sentence_here&level=level_type&target_lang=target_language)

For example, you can translate the sentence "Be nice." by running the following command:

```bash
curl -H 'Content-Type: application/json' -X POST -d '{"level":"word", "sentence":"Be nice.", "target_lang":"chinese"}' http://localhost:5000/translate_eng
```

And the following will be the json response:

```json
{
    "level": "word",
    "sentence": "Be nice.",
    "target_lang": "chinese",
    "translated": "和气点。"
}
```

Here are some examples for eng translation using some other configuration options:

```bash
curl -H 'Content-Type: application/json' -X POST -d '{"level":"char", "sentence":"Be nice.", "target_lang":"chinese"}' http://localhost:5000/translate_eng
curl -H 'Content-Type: application/json' -X POST -d '{"level":"word-glove", "sentence":"Be nice.", "target_lang":"chinese"}' http://localhost:5000/translate_eng
curl -H 'Content-Type: application/json' -X POST -d '{"level":"word", "sentence":"Be nice.", "target_lang":"french"}' http://localhost:5000/translate_eng
curl -H 'Content-Type: application/json' -X POST -d '{"level":"word-glove", "sentence":"Be nice.", "target_lang":"french"}' http://localhost:5000/translate_eng
curl -H 'Content-Type: application/json' -X POST -d '{"level":"char", "sentence":"Be nice.", "target_lang":"french"}' http://localhost:5000/translate_eng
```







