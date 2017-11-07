# keras-language-translator-web-api

A simple language translator implemented in Keras with Flask serving web

The language translator is built based on seq2seq models, and can infer based on either character-level or word-level. 

The seq2seq model is implemented using LSTM encoder-decoder on Keras. 

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

The translator models are chained using eng-to-french and eng-to-chinese data set and are available in the 
translator_train/models directory. During runtime, the flask app will load these trained models to perform the 
translation

Goto translator_web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained seq2seq models:

* Character-level seq2seq models
* Word-level seq2seq models

To translate an english sentence to other languages using web api, after the flask server is started, run the following curl POST query
in your terminal:

```bash
curl -H 'Content-Type application/json' -X POST -d '{"level":"level_type", "sentence":"your_sentence_here", "target_lang":"target_language"}' http://localhost:5000/translate_eng
```

The level_type can be "char" or "word", the target_lang can be "chinese" or "french"

(Note that same results can be obtained by running a curl GET query to http://localhost:5000/translate_eng?sentence=your_sentence_here&level=level_type&target_lang=target_language)

For example, you can get the sentiments for the sentence "i like the Da Vinci Code a lot." by running the following command:

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

Here are some examples to query sentiments using some other neural network models:

```bash
curl -H 'Content-Type: application/json' -X POST -d '{"level":"char", "sentence":"Be nice.", "target_lang":"chinese"}' http://localhost:5000/translate_eng
curl -H 'Content-Type: application/json' -X POST -d '{"level":"word", "sentence":"Be nice.", "target_lang":"french"}' http://localhost:5000/translate_eng
curl -H 'Content-Type: application/json' -X POST -d '{"level":"char", "sentence":"Be nice.", "target_lang":"french"}' http://localhost:5000/translate_eng
```







