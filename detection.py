import streamlit as st # type: ignore
import plotly.express as px
import pandas as pd
import os
import warnings
import numpy as np
import seaborn as sns
import nltk # type: ignore
import matplotlib.pyplot as plt
import re # Library for cleaning
import time
import pickle
import seaborn as sns
import math
import tensorflow as tf # type: ignore
from sklearn.utils import shuffle
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE # type: ignore
from PIL import Image
from re import X
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # type: ignore
from nltk.stem import PorterStemmer # type: ignore
from nltk.stem.snowball import SnowballStemmer # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # type: ignore
from datetime import datetime
import re # Library for cleaning
# !pip install Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # type: ignore
from nltk.stem import PorterStemmer # type: ignore
from nltk.stem.snowball import SnowballStemmer # type: ignore
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
# !pip install googletrans==4.0.0-rc1
from googletrans import Translator # type: ignore
warnings.filterwarnings('ignore')

st.title(" :robot_face: Buzzer and Real User Detection")
st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>',unsafe_allow_html=True)

# Download NLTK resources
nltk.download('stopwords')
stop_words = stopwords.words('indonesian')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('indonesian'))

# Load kamus kata tidak baku
kamus_data = pd.read_excel("lexicon/kamuskatabaku.xlsx")
kamus_tidak_baku = dict(zip(kamus_data['kata_tidak_baku'], kamus_data['kata_baku']))

# Load leksikon INSET
leksikon_df = pd.read_csv("lexicon/lexiconfull.csv")
lexicon_positive = {row['word']: int(row['weight']) for _, row in leksikon_df[leksikon_df['weight'] > 0].iterrows()}
lexicon_negative = {row['word']: int(row['weight']) for _, row in leksikon_df[leksikon_df['weight'] < 0].iterrows()}

# Load the fitted TF-IDF vectorizer
with open("tfidf_vectorizer_inset.pkl", "rb") as file:
    tfidf_inset = pickle.load(file)
with open("tfidf_vectorizer_vader.pkl", "rb") as file:
    tfidf_vader = pickle.load(file)
# Load the Tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Load trained models
with open("ml_models_inset.pkl", "rb") as file:
    models_inset = pickle.load(file)
with open("ml_models_vader.pkl", "rb") as file:
    models_vader = pickle.load(file)
# LSTM
model_lstm = tf.keras.models.load_model("lstm_model.keras")

# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Inisialisasi SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
translator = Translator()

# Function untuk preprocessing
def preprocessing_text(text):
    if not isinstance(text, str):
        return ""
    
    # Cleaning
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Hapus URL
    text = re.sub(r'@[^\s]+', '', text)  # Hapus username
    text = re.sub(r'<.*?>', '', text)  # Hapus HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Hapus simbol
    text = re.sub(r'\d', '', text)  # Hapus angka
    
    # Case Folding
    text = text.lower()
    
    # Normalization
    words = text.split()
    normalized_words = [kamus_tidak_baku[word] if word in kamus_tidak_baku else word for word in words]
    
    # Stopwords Removal
    filtered_tokens = [word for word in normalized_words if word not in stop_words]
    
    # Stemming
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return ' '.join(stemmed_tokens)

# Function untuk translate text ke Inggris (untuk VADER)
def translate_tweet(tweet):
    try:
        translated = translator.translate(tweet, src='id', dest='en')
        return translated.text
    except Exception as e:
        return str(e)

# Function untuk labelling sentimen metode InSet
def lexicon_polarity_indonesia(text):
    score = 0
    for word in text.split():
        score += lexicon_positive.get(word, 0)
        score += lexicon_negative.get(word, 0)
    
    if score > 0:
        polarity = "Positif"
    elif score < 0:
        polarity = "Negatif"
    else:
        polarity = "Netral"
    
    return score, polarity

# Function untuk labelling sentimen metode VADER
def determine_vader_sentiment(text):
    translated_text = translate_tweet(text)  # Terjemahkan setelah preprocessing
    sentiment_scores = sia.polarity_scores(translated_text)
    positive_score = sentiment_scores['pos'] * 100
    negative_score = sentiment_scores['neg'] * 100
    neutral_score = sentiment_scores['neu'] * 100
    compound_score = sentiment_scores['compound'] * 100
    
    if compound_score >= 0.05:
        sentiment = "Positif"
    elif compound_score <= -0.05:
        sentiment = "Negatif"
    else:
        sentiment = "Netral"
    
    return positive_score, negative_score, neutral_score, compound_score, sentiment, translated_text

# Function untuk menentukan apakah user adalah buzzer
def categorize_buzzer(sentiment):
    return "Buzzer" if sentiment in ["Positif", "Negatif"] else "Real User"

def tfidf_vector(text, option):
    if option == "InSet Lexicon":
        text_tfidf = tfidf_inset.transform([text])
    elif option == "VADER Lexicon":
        text_tfidf = tfidf_vader.transform([text])

    text_tfidf_dense = text_tfidf.toarray() if hasattr(text_tfidf, "toarray") else text_tfidf
    
    predictions_text = {}
    if option == "InSet Lexicon":
        for model_name, model in models_inset.items():
            prediction = model.predict(text_tfidf_dense)[0]
            confidence_score = model.predict_proba(text_tfidf_dense).max(axis=1)[0]
            predictions_text[model_name] = (prediction, confidence_score)
    elif option == "VADER Lexicon":
        for model_name, model in models_vader.items():
            prediction = model.predict(text_tfidf_dense)[0]
            confidence_score = model.predict_proba(text_tfidf_dense).max(axis=1)[0]
            predictions_text[model_name] = (prediction, confidence_score)
    
    return predictions_text

def token_lstm(text):
    max_length = 43
    # Convert to sequences
    sequence = tokenizer.texts_to_sequences([text])  # Convert to sequences

    # Padding
    sequence_padded = pad_sequences(sequence, maxlen=max_length, padding="post")

    # Get probability
    # [0]: Mengakses elemen pertama dari array (karena hanya 1 data yang diprediksi).
    # [0]: Mengakses nilai prediksinya (hasil sigmoid) dari dalam nested array.
    prediction = model_lstm.predict(sequence_padded)[0][0]

    if prediction > 0.5:
        return "Buzzer", prediction  # confidence score untuk Buzzer
    else:
        return "Real User", 1 - prediction  # confidence score untuk Real User

# Fungsi untuk preprocessing dan prediksi (menggunakan cache agar tidak dieksekusi ulang)
@st.cache_data
def preprocess_and_predict(text, option, option_algo):
    # Display raw text
    st.markdown("**Raw Text:**")
    st.warning(text)

    with st.spinner("TEXT PREPROCESSING IN PROGRESS!"):
        time.sleep(2)  # Simulasi pemrosesan
        
    # Preprocess text
    text = preprocessing_text(text)

    st.write("**Preprocessed Text:** ")
    if option == "InSet Lexicon":
        st.info(text)
    elif option == "VADER Lexicon":
        pos, neg, neu, comp_score, sentiment, translated_text = determine_vader_sentiment(text)
        st.info(translated_text)
    

    with st.spinner("SENTIMENT PREDICTION IN PROGRESS!"):
        time.sleep(5)

    # Pemilihan metode lexicon
    if option == "InSet Lexicon":
        score, sentiment = lexicon_polarity_indonesia(text)
        category = categorize_buzzer(sentiment)
    elif option == "VADER Lexicon":
        pos, neg, neu, comp_score, sentiment, translated_text = determine_vader_sentiment(text)
        category = categorize_buzzer(sentiment)

    #st.write(f"Debug: category = {category}")
    st.markdown("**Sentiment Outcome:** ")
    # st.info(f"Sentimennya adalah **{sentiment.lower()} ({comp:.2f}%)**")
    st.info(f"Sentimennya adalah **{sentiment.lower()}**")

    with st.spinner("DETECTION IN PROGRESS!"):
        time.sleep(2)

    if option_algo == "Machine Learning (SVM & Random Forest)":
        if option == "InSet Lexicon":
            predictions = tfidf_vector(text, option)
        elif option == "VADER Lexicon":
            predictions = tfidf_vector(translated_text, option)
        return predictions
    elif option_algo == "Deep Learning (LSTM)":
        if option == "InSet Lexicon":
            predictions, conf_score = token_lstm(text)
        elif option == "VADER Lexicon":
            predictions, conf_score = token_lstm(translated_text)
        return predictions, conf_score


# def preprocess_predict_lstm(text, option):
#     # Display raw text
#     st.markdown("**Raw Text:**")
#     st.warning(text)

#     with st.spinner("TEXT PREPROCESSING IN PROGRESS!"):
#         time.sleep(2)  # Simulasi pemrosesan
        
#     # Preprocess text
#     text = preprocessing_text(text)

#     st.write("**Preprocessed Text:** ")
#     if option == "InSet Lexicon":
#         st.info(text)
#     elif option == "VADER Lexicon":
#         translated_text = determine_vader_sentiment(text)
#         st.info(translated_text)

#     with st.spinner("SENTIMENT PREDICTION IN PROGRESS!"):
#         time.sleep(5)

st.markdown("**Using machine learning classification algorithm and deep learning algorithm which are SVM, Random Forest, and LSTM to detect buzzer or real user on X**")
st.image("xbanner.jpg", width=700)

# Instruksi untuk user
st.write("**Instructions:**")
st.info("Enter a text or tweet or sentence into the text box below and then click on the **Detect Now** button to detect buzzer or real user. ")

# Prompt user untuk memasukkan text 
text = st.text_area("Enter a text or tweet or sentence: ", value = "", placeholder = "Enter your text/ tweets here (e.g. Doktif lebih membantu masyarakat ketimbang owner yang jual overclaim demi cuan)")

# Inisialisasi session state untuk tombol klik
if "clicked" not in st.session_state:
    st.session_state.clicked = False

with st.form("settings_form"):
    # Selectbox untuk memilih metode lexicon
    option = st.selectbox(
        "Choose Lexicon Method:",
        ("InSet Lexicon", "VADER Lexicon"),
        index=None,
        placeholder="Select lexicon method..."
    )

    # Selectbox untuk memilih model ML atau DL
    option_algo = st.selectbox(
        "Choose Algorithm:",
        ("Machine Learning (SVM & Random Forest)", "Deep Learning (LSTM)"),
        index=None,
        placeholder="Select algorithm...")
    
    # Prompt user untuk klik submit btn
    detect_btn = st.form_submit_button("Detect Now", type="primary")


# Tombol untuk memproses teks
if detect_btn or st.session_state.clicked:
    st.session_state.clicked = True  # Menyimpan status tombol

    if text.strip() == "":
        st.error("Error! Input text or tweet can't be empty.")
    elif option is None:
        st.warning("Please choose lexicon method before processing.")
    else:
        if option_algo is None:
            st.warning("Please choose algorithm before processing.")
        elif option_algo == "Machine Learning (SVM & Random Forest)":
            # Memanggil fungsi preprocessing dan prediksi
            predictions = preprocess_and_predict(text, option, option_algo)
            
            # Menampilkan hasil prediksi
            st.markdown("**Detection Outcome:** ")

            for model, (pred, conf_score) in predictions.items():
                if model == "Random Forest" or model == "Support Vector Machine":  # Menampilkan untuk model Random Forest/SVM
                    if pred == 'Buzzer':
                        st.error(f"{model}: terdeteksi **{pred.lower()} ({conf_score:.2%})**.")
                    elif pred == 'Real User':
                        st.success(f"{model}: terdeteksi **{pred.lower()} ({conf_score:.2%})**.")

                    # Pie Chart untuk distribusi confidence score
                    st.markdown(f"**{model} Confidence Distribution:**")
                    labels = ["Buzzer", "Real User"]
                    if pred == "Buzzer":
                        values = [round(conf_score * 100, 2), round((1 - conf_score) * 100, 2)]
                    else:
                        values = [round((1 - conf_score) * 100, 2), round(conf_score * 100, 2)]

                    fig = px.pie(
                        names = labels,
                        values = values,
                        color=labels,
                        color_discrete_map={"Buzzer": "#8F2700", "Real User": "#49734A"},
                        hole = 0.4
                    )
                    fig.update_traces(textinfo='label+percent', pull=[0.05, 0], textfont_color='white', textposition='outside', textfont_size=16)
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

        elif option_algo == "Deep Learning (LSTM)":
            # Memanggil fungsi preprocessing dan prediksi
            predictions, conf_score = preprocess_and_predict(text, option, option_algo)

            # Menampilkan hasil prediksi
            st.markdown("**Detection Outcome:** ")

            if predictions == 'Buzzer':
                st.error(f"LSTM: terdeteksi **{predictions.lower()} ({conf_score:.2%})**.")
            elif predictions == 'Real User':
                st.success(f"LSTM: terdeteksi **{predictions.lower()} ({conf_score:.2%})**.")

            # Pie Chart untuk distribusi confidence score
            st.markdown("**Confidence Distribution:**")

            labels = ["Buzzer", "Real User"]

            # Atur values berdasarkan hasil prediksi
            if predictions == "Buzzer":
                values = [round(conf_score * 100, 2), round((1 - conf_score) * 100, 2)]
            else:
                values = [round((1 - conf_score) * 100, 2), round(conf_score * 100, 2)]

            fig = px.pie(
                names=labels,
                values=values,
                color=labels,
                color_discrete_map={"Buzzer": "#8F2700", "Real User": "#49734A"},
                hole=0.4
            )
            fig.update_traces(
                textinfo='label+percent',
                pull=[0.05 if label == predictions else 0 for label in labels],
                textfont_color='white',
                textposition='outside',
                textfont_size=16
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)



