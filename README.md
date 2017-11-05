# keras-language-translator-web-api

A simple language translator implemented in Keras with Flask serving web

The language translator is built based on seq2seq models, and can infer based on either character-level or word-level. 

The seq2seq model is implemented using LSTM encoder-decoder on Keras. 

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

Goto translator_web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained seq2seq models:

* Character-level seq2seq models
* Word-level seq2seq models

