from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import random
import nltk
from flask_cors import CORS
from flask import abort, Flask, jsonify, redirect, request, url_for
f = open('data.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()
sentence_list = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
# Salam Pembuka
salam_input_user = ("hello", "hi", "yuhuy", "oiy", "hey")
salam_respon_bot = ["hi", "hello", "yuhuy", "oit", "hey"]

def greeting(scentence):

    for word in scentence.split():
        if word.lower() in salam_input_user:
            return random.choice(salam_respon_bot)
# Respon
def response(user_response):
    chatbot_response = ''
    sentence_list.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sentence_list)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        chatbot_response = chatbot_response + "Maaf, kami tidak bisa mengerti pertanyaan anda"
        return chatbot_response

    else:
        chatbot_response = chatbot_response+sentence_list[idx]
        return chatbot_response

def response_api(data):
    return (
        jsonify(**data),
        data['code']
    )
app = Flask(__name__)
CORS(app)
@app.route("/")
def hello_world():
    return "<p>Hello, World! Deploy nich...</p>"
@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        data = request.form['message']
        hasil=response(data)
        if(data==hasil):
            hasil="Maaf, kami tidak bisa mengerti pertanyaan anda"
        return response_api({
            'code': 200,
            'message': 'Berhasil',
            'data': hasil
        })
    else:
        return response_api({
            'code': 400,
            'message': 'Gagal',
            'data': 'Tidak dapat mengakses'
        })