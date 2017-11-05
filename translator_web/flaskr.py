from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for
from translator_web.eng_to_fra_char_translator_predict import EngToFraCharTranslator
from translator_web.eng_to_cmn_char_translator_predict import EngToCmnCharTranslator
from translator_web.eng_to_fra_word_translator_predict import EngToFraWordTranslator
from translator_web.eng_to_cmn_word_translator_predict import EngToCmnWordTranslator

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

eng_to_fra_translator_c = EngToFraCharTranslator()
eng_to_fra_translator_c.test_run()

eng_to_cmn_translator_c = EngToCmnCharTranslator()
eng_to_cmn_translator_c.test_run()

eng_to_fra_translator_w = EngToFraWordTranslator()
eng_to_fra_translator_w.test_run()

eng_to_cmn_translator_w = EngToCmnWordTranslator()
eng_to_cmn_translator_w.test_run()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/eng_to_fra_char_translator', methods=['POST', 'GET'])
def eng_to_fra_char_translator():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            translated = eng_to_fra_translator_c.translate_lang(sent)
            return render_template('eng_to_fra_char_translator_result.html', sentence=sent, translated=translated)
    return render_template('eng_to_fra_char_translator.html')


@app.route('/eng_to_cmn_char_translator', methods=['POST', 'GET'])
def eng_to_cmn_char_translator():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            translated = eng_to_fra_translator_c.translate_lang(sent)
            return render_template('eng_to_cmn_char_translator_result.html', sentence=sent,
                                   translated=translated)
    return render_template('eng_to_cmn_char_translator.html')


@app.route('/eng_to_fra_word_translator', methods=['POST', 'GET'])
def eng_to_fra_word_translator():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            translated = eng_to_fra_translator_w.translate_lang(sent)
            return render_template('eng_to_fra_word_translator_result.html', sentence=sent,
                                   translated=translated)
    return render_template('eng_to_fra_word_translator.html')


@app.route('/eng_to_cmn_word_translator', methods=['POST', 'GET'])
def eng_to_cmn_word_translator():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = eng_to_cmn_translator_w.translate_lang(sent)
            return render_template('eng_to_cmn_word_translator_result.html', sentence=sent,
                                   sentiments=sentiments)
    return render_template('eng_to_cmn_word_translator.html')


if __name__ == '__main__':
    app.run(debug=True)
