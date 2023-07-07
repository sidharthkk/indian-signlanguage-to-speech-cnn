from flask import Flask, render_template
import subprocess
import numpy as np
import signal
import sys
import gtts
from playsound import playsound


app = Flask(__name__)

letter_process = None
words_process = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_letter')
def run_letter():
    t1 = gtts.gTTS("Place your hand towards the camera module.")
    soundName=str(np.random.randint(0,100))+".mp3"
    t1.save(soundName)
    playsound(soundName)
    global letter_process
    if letter_process is None or letter_process.poll() is not None:
        letter_process = subprocess.Popen([sys.executable, 'letters.py'])
        return 'Letter.py is running'
    else:
        return 'Letter.py is already running'


@app.route('/run_words')
def run_words():
    global words_process
    if words_process is None or words_process.poll() is not None:
        words_process = subprocess.Popen([sys.executable, 'words.py'])
        return 'Words.py is running'
    else:
        return 'Words.py is already running'


@app.route('/clear_program')
def clear_program():
    global letter_process, words_process
    if letter_process is not None:
        letter_process.kill()
        letter_process = None
    if words_process is not None:
        words_process.kill()
        words_process = None
    return 'Program terminated'


if __name__ == '__main__':
    app.run()
