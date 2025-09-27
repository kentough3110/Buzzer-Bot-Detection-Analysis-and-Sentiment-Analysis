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

import json
import requests  # type: ignore
import streamlit as st  # type: ignore
from streamlit_lottie import st_lottie  # type: ignore

# GitHub: https://github.com/andfanilo/streamlit-lottie
# Lottie Files: https://lottiefiles.com/

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.title(" 📢 Introduction")
st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>',unsafe_allow_html=True)

lottie_intro = load_lottiefile("assets/intro.json")
col1, col2 = st.columns((2))
with col1:
    st_lottie(
        lottie_intro,
        speed= 1,
        loop= True,
        height= 600,
        width= 600,
        key= "thinking"
    )
with col2:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.subheader("**Wassup, I'm Kent 👋**")
    st.header("Aspiring Data Analyst from Indonesia")
    st.write(
    "This mini-app provides a comprehensive analysis of buzzer activities and includes a detection tool designed to identify buzzer towards coordinated disinformation campaigns on Platform X. "
    "By leveraging data analytics and visualization, it helps users: "
    )
    st.write("📍 **Leverages data analytics to uncover buzzer activity patterns.**")
    st.write("📍 **Utilizes visualizations to enhance understanding of disinformation spread.**")
    st.write("📍 **Identifies and monitors coordinated disinformation campaigns on Platform X.**")
    st.write("📍 **Supports efforts to maintain information integrity online.**")
st.divider()
lottie_intro2 = load_lottiefile(r"C:\Users\KENT LEE\Documents\TA KENT 2025\assets\intro2.json")
col1, col2 = st.columns((2))
with col1:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.subheader("**Analysis Method 📊**")
    st.header("Overview of Buzzer Detection Techniques")
    st.write(
    "By leveraging data analytics and modeling techniques, this analysis applies the following methods:"
    )
    st.write("📍 **Lexicon-based sentiment analysis (VADER-EN and InSet-ID)**")
    st.write("📍 **TF-IDF and classic machine learning classification (SVM, GBM, Random Forest, Naïve Bayes)**")
    st.write("📍 **Sequence padding and Long Short-Term Memory (LSTM)**")
with col2:
    st_lottie(
    lottie_intro2,
    speed= 1,
    loop= True,
    height= 600,
    width= 600,
    key= "robot"
)
