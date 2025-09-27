import streamlit as st # type: ignore
import plotly.express as px
import pandas as pd
import os
import warnings
import numpy as np
import seaborn as sns
import nltk # type: ignore
import matplotlib.pyplot as plt
import re
import time
import math
import tensorflow as tf # type: ignore
import io
import sys
from tensorflow.keras.utils import plot_model # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
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
from nltk.sentiment.vader import SentimentIntensityAnalyzer # type: ignore
nltk.download('stopwords')
stop_words = stopwords.words('indonesian')
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # type: ignore
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
warnings.filterwarnings('ignore')

# Fungsi-fungsi utilitas
def load_data(filename=None):
    if filename is not None:
        df = pd.read_excel(filename)
    else:
        os.chdir(r"C:\Users\KENT LEE\Documents\TA KENT 2025")
        df = pd.read_excel("doktifall.xlsx")
    return df

def preprocess_datetime(df):
    for index, row in df.iterrows():
        try:
            date_time_obj = datetime.strptime(row['created_at'], '%a %b %d %H:%M:%S %z %Y')
            df.at[index, 'tanggal'] = date_time_obj.strftime('%a %b %d')
            df.at[index, 'waktu'] = date_time_obj.strftime('%H:%M:%S')
        except ValueError:
            st.write(f"Done: data berhasil di bagi: {row['created_at']}")
    return df[['tanggal','waktu','username','favorite_count','quote_count','reply_count','retweet_count','full_text']]

def plot_word_frequency(text, title):
    tokens = text.split()
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    word, count = zip(*top_words)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(word, count, color=plt.cm.tab10(range(len(word))))
    ax.set_xlabel("Kata yang Sering Muncul", fontsize=12, fontweight='bold')
    ax.set_ylabel("Jumlah Kata", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xticks(range(len(word)))
    ax.set_xticklabels(word, rotation=45)
    
    for bar, num in zip(bars, count):
        ax.text(bar.get_x() + bar.get_width() / 2, num + 0.5, str(num), fontsize=12, color='black', ha='center')
    
    return fig

def generate_wordcloud(text, title, stopwords=None):
    if stopwords is None:
        stopwords = set(STOPWORDS)
        stopwords.update(['https', 'co', 'RT', '...', 'amp'])
    
    wc = WordCloud(stopwords=stopwords, background_color="white", max_words=500, width=800, height=450)
    wc.generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    return fig

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'['
        u'\U0001F600-\U0001F64F'
        u'\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF'
        u'\U0001F700-\U0001F77F'
        u'\U0001F780-\U0001F7FF'
        u'\U0001F800-\U0001F8FF'
        u'\U0001F900-\U0001F9FF'
        u'\U0001FA00-\U0001FA6F'
        u'\U0001FA70-\U0001FAFF'
        u'\U0001F004-\U0001F0CF'
        u'\U0001F1E0-\U0001F1FF'
        ']+', '', text, flags=re.UNICODE)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\d', '', text)
    return text

def case_folding(text):
    if isinstance(text, str):
        return text.lower()
    return text

def normalize_text(text, kamus_tidak_baku):
    if not isinstance(text, str):
        return '', [], [], []
    
    replaced_words = []
    for word in text.split():
        if word in kamus_tidak_baku:
            baku_word = kamus_tidak_baku[word]
            if isinstance(baku_word, str) and baku_word.isalpha():
                replaced_words.append(baku_word)
        else:
            replaced_words.append(word)
    
    return ' '.join(replaced_words)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def plot_sentiment_distribution(df, column, title):
    sentiment_count = df[column].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette='tab10', ax=ax)
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Class Sentiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count Tweet', fontsize=12, fontweight='bold')
    
    total = sentiment_count.sum()
    for i, count in enumerate(sentiment_count.values):
        percentage = f"{100 * count / total:.2f}%"
        ax.text(i, count + 0.10, f"{count}\n({percentage})", ha='center', va='bottom')
    
    return fig

def plot_category_distribution(df, column, title):
    counts = df[column].value_counts()
    total = counts.sum()
    percentages = (counts / total) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(counts.index, counts.values, color=['tab:blue', 'tab:orange'])
    
    for bar, count, percentage in zip(bars, counts.values, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, count,
                f"{count}\n({percentage:.1f}%)", ha='center', va='bottom', fontsize=12)
    
    ax.set_xlabel("Category", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    
    return fig

# Fungsi-fungsi utama
def show_raw_data(df):
    st.header("Raw Tweet Dataset", anchor=False, divider="blue")
    df = preprocess_datetime(df)
    st.write(df.head(5))
    return df

def show_eda_before_preprocessing(df):
    st.header("EDA Before Preprocessing", divider="blue")
    col1, col2 = st.columns((2))
    
    with col1:
        st.subheader("Words Frequency Distribution")
        text = " ".join(df["full_text"])
        fig = plot_word_frequency(text, "Frekuensi Kata")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Wordcloud Dataset")
        text = ' '.join(df['full_text'].astype(str).fillna('').tolist())
        fig = generate_wordcloud(text, "Wordcloud Dataset Sebelum Preprocessing")
        st.pyplot(fig)

def show_preprocessing(df):
    st.header("Preprocessing", divider="blue")
    st.subheader("Data Cleaning")
    df['cleaning'] = df['full_text'].apply(clean_text)
    st.write(df[['full_text', 'cleaning']].head(5))
    
    st.header("NLP Pipeline", divider="blue")
    
    st.subheader("Case Folding")
    df['case_folding'] = df['cleaning'].apply(case_folding)
    st.write(df[['cleaning', 'case_folding']].head(5))
    
    st.subheader("Normalization")  
    kamus_data = pd.read_excel(r"C:\Users\KENT LEE\Documents\TA KENT 2025\lexicon\kamuskatabaku.xlsx")
    kamus_tidak_baku = dict(zip(kamus_data['kata_tidak_baku'], kamus_data['kata_baku']))
    df['normalization'] = df['case_folding'].apply(lambda x: normalize_text(x, kamus_tidak_baku))
    st.write(df[['case_folding', 'normalization']].head(5))
    
    st.subheader("Tokenization")  
    df['tokenization'] = df['normalization'].str.split()
    st.write(df[['normalization', 'tokenization']].head(5))
    
    st.subheader("Stopwords Removal")  
    df['stopword_removal'] = df['tokenization'].apply(remove_stopwords)
    st.write(df[['tokenization', 'stopword_removal']].head(5))
    
    st.subheader("Stemming")  
    df = pd.read_csv(r"C:\Users\KENT LEE\Documents\TA KENT 2025\doktif_after_preprocessing.csv", encoding="ISO-8859-1")
    st.write(df[['stopword_removal', 'stemming_data']].head(5))

    st.subheader("English Translation")  
    df = pd.read_csv(r"C:\Users\KENT LEE\Documents\TA KENT 2025\doktif_after_preprocessing.csv", encoding="ISO-8859-1")
    st.write(df[['stemming_data', 'eng_translation']].head(5))
    
    return df

def show_eda_after_preprocessing(df):
    st.header("EDA After Preprocessing", divider="blue")
    col1, col2 = st.columns((2))
    
    with col1:
        st.subheader("Words Frequency Distribution")
        text = " ".join(df['stemming_data'].astype(str).fillna(''))
        fig = plot_word_frequency(text, "Frekuensi Kata Setelah Preprocessing")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Wordcloud Dataset")
        text = ' '.join(df['stemming_data'].astype(str).fillna('').tolist())
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(['https', 'co', 'RT', '...', 'amp', 'ya', 'lu', 'gue', 'sih', 'sama', 'nih', 'tuh', 'bikin', 'kak', 'dapat', 'kayak'])
        fig = generate_wordcloud(text, "Wordcloud Dataset Setelah Preprocessing", custom_stopwords)
        st.pyplot(fig)

def show_inset_lexicon_analysis(df):
    st.header("InSet Lexicon", divider='blue') 
    st.subheader("Labelling") 
    
    leksikon_df = pd.read_csv(r"C:\Users\KENT LEE\Documents\TA KENT 2025\lexicon\lexiconfull.csv")
    lexicon_positive = {row['word']: int(row['weight']) for _, row in leksikon_df[leksikon_df['weight'] > 0].iterrows()}
    lexicon_negative = {row['word']: int(row['weight']) for _, row in leksikon_df[leksikon_df['weight'] < 0].iterrows()}

    def lexicon_polarity_indonesia(text):
        score = 0
        if isinstance(text, str):
            for word in text.split():
                if word in lexicon_positive:
                    score += lexicon_positive[word]
                elif word in lexicon_negative:
                    score += lexicon_negative[word]
        
        if score > 0:
            polarity = "Positif"
        elif score < 0:
            polarity = "Negatif"
        else:
            polarity = "Netral"
        
        return score, polarity

    df[['Score', 'Sentiment']] = df['stemming_data'].apply(lambda x: pd.Series(lexicon_polarity_indonesia(x)))
    df['Category'] = df['Sentiment'].apply(lambda x: "Buzzer" if x in ["Positif", "Negatif"] else "Real User")
    st.write(df[['stemming_data', 'Score', 'Sentiment', 'Category']].head(10))

    col1, col2 = st.columns((2))
    with col1:
        st.subheader("Sentiment Analysis Distribution")
        fig = plot_sentiment_distribution(df, 'Sentiment', 'Jumlah Analisis Sentimen Metode InSet Lexicon')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Buzzer vs Real User Distribution")
        fig = plot_category_distribution(df, 'Category', 'Jumlah Buzzer vs Real User Metode InSet Lexicon')
        st.pyplot(fig)
    
    return df

def show_vader_lexicon_analysis(df):
    st.header("VADER Lexicon", divider='blue')
    st.subheader("Labelling")
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    def determine_vader_sentiment(text):
        if isinstance(text, str):
            sentiment_scores = sia.polarity_scores(text)
            compound_score = sentiment_scores['compound']
            
            if compound_score >= 0.05:
                sentiment = "Positif"
            elif compound_score <= -0.05:
                sentiment = "Negatif"
            else:
                sentiment = "Netral"
            
            return sentiment_scores['pos'], sentiment_scores['neg'], sentiment_scores['neu'], compound_score, sentiment
        return 0, 0, 0, 0, "Netral"

    df[['Positive', 'Negative', 'Neutral', 'Compound Score', 'Sentiment']] = df['eng_translation'].apply(lambda x: pd.Series(determine_vader_sentiment(x)))
    df['Category'] = df['Sentiment'].apply(lambda x: "Buzzer" if x in ["Positif", "Negatif"] else "Real User")
    st.write(df[['eng_translation', 'Positive', 'Negative', 'Neutral', 'Compound Score', 'Sentiment', 'Category']].head(10))

    col1, col2 = st.columns((2))
    with col1:
        st.subheader("Sentiment Analysis Distribution")
        fig = plot_sentiment_distribution(df, 'Sentiment', 'Jumlah Analisis Sentimen Metode VADER Sentiment')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Buzzer vs Real User Distribution")
        fig = plot_category_distribution(df, 'Category', 'Jumlah Buzzer vs Real User Metode VADER Lexicon')
        st.pyplot(fig)
    
    return df

def train_test_split_data(x, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def plot_train_test_distribution(X_train, X_test):
    dtrain = len(X_train)
    dtest = len(X_test)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Data Training', 'Data Testing'], [dtrain, dtest], 
                color=['tab:blue', 'tab:orange'])
    
    for bar, count in zip(bars, [dtrain, dtest]):
        percentage = f"{count / (dtrain + dtest) * 100:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, 
                f"{count}\n({percentage})", ha='center', va='bottom', fontsize=12)
    
    ax.set_xlabel("Kategori Data", fontsize=12, fontweight='bold')
    ax.set_ylabel("Jumlah Data", fontsize=12, fontweight='bold')
    ax.set_title("Jumlah Data Training dan Data Testing", fontsize=14, pad=15, fontweight='bold')
    
    return fig

def tfidf_vectorize(X_train, X_test, max_features=2000, min_df=5, max_df=0.7, 
                    ngram_range=(1,1), stop_words=None):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words
    )
    x_train_tfidf = tfidf.fit_transform(X_train)
    x_test_tfidf = tfidf.transform(X_test)
    return x_train_tfidf, x_test_tfidf, tfidf

def apply_smote(x_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
    return x_resampled, y_resampled

def plot_class_distribution(y_before, y_after, title_before, title_after):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before
    counts_before = y_before.value_counts()
    axes[0].bar(counts_before.index.astype(str), counts_before.values, color='tab:blue')
    axes[0].set_title(title_before, fontweight='bold')
    axes[0].set_xlabel("Kategori", fontweight='bold')
    axes[0].set_ylabel("Jumlah", fontweight='bold')
    
    # After
    counts_after = pd.Series(y_after).value_counts()
    axes[1].bar(counts_after.index.astype(str), counts_after.values, color='tab:orange')
    axes[1].set_title(title_after, fontweight='bold')
    axes[1].set_xlabel("Kategori", fontweight='bold')
    axes[1].set_ylabel("Jumlah", fontweight='bold')
    
    # Add labels
    for ax, counts in zip(axes, [counts_before, counts_after]):
        for i, v in enumerate(counts.values):
            ax.text(i, v + 10, str(v), ha='center', fontsize=12, color='black')
    
    plt.tight_layout()
    return fig

def initialize_ml_models():
    return {
        "Support Vector Machine": SVC(kernel='linear', random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting Machine (GBM)": GradientBoostingClassifier(random_state=42),
    }

def train_evaluate_models(models, x_train, y_train, x_test, y_test):
    results = {}
    predictions = {}
    
    for model_name, model in models.items():
        # Convert sparse matrix to dense array if needed
        x_train_dense = x_train.toarray() if hasattr(x_train, "toarray") else x_train
        x_test_dense = x_test.toarray() if hasattr(x_test, "toarray") else x_test
        
        # Train model
        start_time = time.time()
        model.fit(x_train_dense, y_train)
        training_time = time.time() - start_time
        minutes, seconds = divmod(int(training_time), 60)
        formatted_time = f"{minutes:02d} m {seconds:02d} s"
        
        # Predictions
        y_pred = model.predict(x_test_dense)
        predictions[model_name] = y_pred
        
        # Store results
        results[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "training_time": formatted_time
        }
    
    return results, predictions

def plot_metric_comparison(before_results, after_results, metric_name):
    models = list(before_results.keys())
    before_values = [before_results[model][metric_name] for model in models]
    after_values = [after_results[model][metric_name] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, before_values, width, label='Before SMOTE', color='tab:blue')
    bars2 = ax.bar(x + width/2, after_values, width, label='After SMOTE', color='tab:orange')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel("Model", fontweight='bold')
    ax.set_ylabel(metric_name.capitalize(), fontweight='bold')
    ax.set_title(f"Perbandingan {metric_name.capitalize()} Model Sebelum dan Sesudah SMOTE", 
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim(0, 1)
    plt.tight_layout()
    
    return fig

def plot_confusion_matrices(results, figsize=(5, 5)):
    # Create a list of models to process
    models = list(results.items())
    
    # Process two models at a time
    for i in range(0, len(models), 2):
        # Create columns for each row
        cols = st.columns(2)
        
        for j in range(2):
            if i + j < len(models):
                model_name, result = models[i + j]
                
                with cols[j]:
                    fig, ax = plt.subplots(figsize=figsize)
                    sns.heatmap(
                        result["confusion_matrix"], 
                        annot=True, 
                        fmt='d', 
                        cbar=False,
                        xticklabels=['Buzzer', 'Real User'],
                        yticklabels=['Buzzer', 'Real User'],
                        ax=ax, 
                        square=True, 
                        linewidth=0.5, 
                        cmap="Blues"
                    )
                    ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
                    ax.set_xlabel("Predicted", fontsize=10)
                    ax.set_ylabel("Actual", fontsize=10)
                    st.pyplot(fig)
                    # plt.close(fig)

def show_classification_reports(results):
    # Convert results dictionary to list of tuples
    models = list(results.items())
    
    # Process two models at a time
    for i in range(0, len(models), 2):
        # Create columns for each row
        cols = st.columns(2)
        
        for j in range(2):
            if i + j < len(models):
                model_name, result = models[i + j]
                
                with cols[j]:
                    with st.container():
                        st.subheader(f"📊 {model_name}")
                        report_df = pd.DataFrame(result['classification_report']).transpose()
                        
                        styled_df = report_df.style.background_gradient(
                            cmap="Blues",
                            subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']],
                            vmin=-1,
                            vmax=1
                        ).format(precision=3)
                        
                        st.dataframe(styled_df)


def build_lstm_model(input_size, embed_dim, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=input_size, output_dim=embed_dim, input_length=max_length))
    model.add(LSTM(64, dropout=0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    return model

def plot_lstm_performance(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Accuracy
    axes[0].plot(history.history['accuracy'], marker='o')
    axes[0].plot(history.history['val_accuracy'], marker='s')
    axes[0].set_title('Model Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_xticks(range(1, len(history.history['accuracy']) + 1))
    axes[0].legend(['Train', 'Validation'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Plot Loss
    axes[1].plot(history.history['loss'], marker='o')
    axes[1].plot(history.history['val_loss'], marker='s')
    axes[1].set_title('Model Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Loss', fontweight='bold')
    axes[1].set_xticks(range(1, len(history.history['loss']) + 1))
    axes[1].legend(['Train', 'Validation'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    return fig

# def evaluate_lstm_model(model, X_test, y_test_encoded, y):
#     y_pred = model.predict(X_test)
#     y_pred_classes = (y_pred > 0.5).astype(int)
    
#     accuracy = accuracy_score(y_test_encoded, y_pred_classes)
#     recall = recall_score(y_test_encoded, y_pred_classes)
#     precision = precision_score(y_test_encoded, y_pred_classes)
#     f1 = f1_score(y_test_encoded, y_pred_classes)
    
#     # Konversi label unik menjadi string
#     target_names = [str(label) for label in np.unique(y)]

#     report = classification_report(y_test_encoded, y_pred_classes, target_names=target_names, output_dict=True)

#     return {
#         "accuracy": accuracy,
#         "recall": report["weighted avg"]["recall"],
#         "precision": report["weighted avg"]["precision"],
#         "f1_score": report["weighted avg"]["f1-score"],
#         "confusion_matrix": confusion_matrix(y_test_encoded, y_pred_classes),
#         "classification_report": report
#     }

def evaluate_lstm_model(model, X_test, y_test_encoded, y):
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test_encoded, y_pred_classes)

    # Buat classification report sebagai dict
    target_names = [str(label) for label in np.unique(y)]
    report = classification_report(
        y_test_encoded,
        y_pred_classes,
        target_names=target_names,
        output_dict=True
    )

    # Ambil nilai dari weighted avg
    weighted_precision = report["weighted avg"]["precision"]
    weighted_recall = report["weighted avg"]["recall"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    return {
        "accuracy": accuracy,
        "recall": weighted_recall,
        "precision": weighted_precision,
        "f1_score": weighted_f1,
        "confusion_matrix": confusion_matrix(y_test_encoded, y_pred_classes),
        "classification_report": report
    }

def show_prediction_comparison(X_test, y_test, predictions, models_to_show=["Random Forest", "Support Vector Machine"]):
    dfTest = pd.DataFrame({"Tweet": X_test, "Category (Data Test)": y_test})
    
    for model_name in models_to_show:
        if model_name in predictions:
            dfTest[f"{model_name} Prediction"] = predictions[model_name]
    
    st.write(dfTest.head(20))

# Main Streamlit App
def main():
    st.title(" :bar_chart: Data Analysis")
    st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>', unsafe_allow_html=True)

    counter = 0
    # fl = st.file_uploader(":file_folder: Upload a file", type=(["csv","txt","xlsx","xls"]))
    # if fl is not None:
    #     filename = fl.name
    #     st.write(filename)
    #     df = pd.read_excel(filename)
    #     counter += 1
    # else:
    #     df = load_data()
    df = load_data()
    if counter == 0:
        df = show_raw_data(df)
        show_eda_before_preprocessing(df)
        df = show_preprocessing(df)
        show_eda_after_preprocessing(df)
        
        df = show_inset_lexicon_analysis(df)
        df = show_vader_lexicon_analysis(df)
        
        # Form untuk memilih metode dan algoritma
        with st.form("settings_form"):
            option = st.selectbox(
                "Choose Lexicon Methodology:",
                ("InSet Lexicon", "VADER Lexicon"),
                index=None,
                placeholder="Select lexicon method..."
            )

            option_algo = st.selectbox(
                "Choose Algorithm:",
                ("Machine Learning (SVM, Random Forest, GBM & Naive Bayes)", "Deep Learning (LSTM)"),
                index=None,
                placeholder="Select algorithm..."
            )

            run_btn = st.form_submit_button("Run", type="primary")

        if run_btn:
            if option == "InSet Lexicon":
                df = df.dropna(subset=['stemming_data'])
                x = df['stemming_data']
                y = df['Category']
            elif option == "VADER Lexicon":
                df = df.dropna(subset=['eng_translation'])
                x = df['eng_translation']
                y = df['Category']
        
            # Split data
            X_train, X_test, y_train, y_test = train_test_split_data(x, y)
            
            st.header("Train Test Split (80:20)", divider='blue')
            col1, col2 = st.columns((2))
            with col1:
                fig = plot_train_test_distribution(X_train, X_test)
                st.pyplot(fig)
            
            st.header("Data Transformation", divider='blue')
            
            if option_algo == "Machine Learning (SVM, Random Forest, GBM & Naive Bayes)":
                st.subheader("TF-IDF Vectorizing")
                stop_words_to_use = stop_words if option == "InSet Lexicon" else 'english'
                x_train_tfidf, x_test_tfidf, tfidf = tfidf_vectorize(
                    X_train, X_test, stop_words=stop_words_to_use
                )
                
                # Show top TF-IDF features
                tfidf_data = tfidf.fit_transform(x)
                tfidf_df = pd.DataFrame(tfidf_data.toarray(), columns=tfidf.get_feature_names_out())
                top_n = 10
                top_tfidf_words = tfidf_df.apply(lambda x: x.nlargest(top_n).index.tolist(), axis=1)
                top_tfidf_values = tfidf_df.apply(lambda x: x.nlargest(top_n).values.tolist(), axis=1)
                top_tfidf_df = pd.DataFrame({"Top Words": top_tfidf_words, "Top Values": top_tfidf_values})
                st.write(top_tfidf_df.head(10))
                
                st.header("Handling Imbalance Class", divider='blue')
                st.subheader("Class Distribution Before vs After SMOTE")
                
                # Before SMOTE
                class_dist_before = y_train.value_counts()
                
                # Apply SMOTE
                x_train_resampled, y_train_resampled = apply_smote(x_train_tfidf, y_train)
                
                # After SMOTE
                class_dist_after = pd.Series(y_train_resampled).value_counts()
                
                fig = plot_class_distribution(
                    y_train, y_train_resampled,
                    "Distribusi Kelas Sebelum SMOTE",
                    "Distribusi Kelas Setelah SMOTE"
                )
                st.pyplot(fig)
                
                # Initialize and train models
                models = initialize_ml_models()
                
                # Before SMOTE
                before_results, _ = train_evaluate_models(models, x_train_tfidf, y_train, x_test_tfidf, y_test)
                
                # After SMOTE
                after_results, predictions = train_evaluate_models(models, x_train_resampled, y_train_resampled, x_test_tfidf, y_test)
                
                st.header("Evaluation & Testing", divider='blue')
                st.subheader("Parameter Comparison for Model")
                col1, col2 = st.columns((2))
                
                with col1:
                    st.subheader("Accuracy Comparison Model Before vs After SMOTE")
                    fig = plot_metric_comparison(before_results, after_results, "accuracy")
                    st.pyplot(fig)
                    
                with col2:
                    st.subheader("F1-Score Comparison Model Before vs After SMOTE")
                    fig = plot_metric_comparison(before_results, after_results, "f1_score")
                    st.pyplot(fig)
                
                st.subheader("Confusion Matrix Setelah SMOTE")
                plot_confusion_matrices(after_results)
                
                st.header("Classification Report")
                show_classification_reports(after_results)
                
                st.header("Actual vs Prediction Comparison")
                show_prediction_comparison(X_test, y_test, predictions)
                
            elif option_algo == "Deep Learning (LSTM)":
                st.subheader("Sequences Conversion and Padding")
                
                # Prepare data for LSTM
                train = pd.DataFrame({'text': X_train, 'Category': y_train})
                test = pd.DataFrame({'text': X_test, 'Category': y_test})
                
                # ✅ Distribusi kelas sebelum upsampling sebagai tabel Streamlit
                class_distribution_before_df = pd.DataFrame({
                'Set': ['Training', 'Training'],
                'Category': ['Buzzer', 'Real User'],
                'Count': [
                    (train['Category'] == 'Buzzer').sum(),
                    (train['Category'] == 'Real User').sum(),
                    ]
                })
                st.markdown("**Distribusi Kelas (Data Train) Sebelum Upsampling:**")
                st.table(class_distribution_before_df)

                # Class distribution before upsampling
                class_dist_before = y_train.value_counts()
                
                # Upsample minority class
                data_majority = train[train['Category'] == 'Buzzer']
                data_minority = train[train['Category'] == 'Real User']
                data_minority_upsampled = resample(
                    data_minority,
                    replace=True,
                    n_samples=data_majority.shape[0],
                    random_state=123
                )
                data_upsampled = pd.concat([data_majority, data_minority_upsampled])
                data_upsampled = shuffle(data_upsampled, random_state=42)
                
                # ✅ Distribusi kelas setelah upsampling sebagai tabel Streamlit
                class_distribution_after_df = pd.DataFrame({
                'Set': ['Training', 'Training'],
                'Category': ['Buzzer', 'Real User'],
                'Count': [
                    (data_upsampled['Category'] == 'Buzzer').sum(),
                    (data_upsampled['Category'] == 'Real User').sum(),
                    ]
                })
                st.markdown("**Distribusi Kelas (Data Train) Setelah Upsampling:**")
                st.table(class_distribution_after_df)

                # Tokenization
                vocab_size = 5000
                tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
                tokenizer.fit_on_texts(x.values)
                
                # Convert text to sequences
                X_train_sequences = tokenizer.texts_to_sequences(data_upsampled['text'].values)
                X_test_sequences = tokenizer.texts_to_sequences(X_test.values)
                
                # Pad sequences
                max_length = max([len(seq) for seq in X_train_sequences])
                X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length)
                X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length)
                
                # Encode labels
                y_train_encoded = pd.get_dummies(data_upsampled['Category']).values[:, 1]
                y_test_encoded = pd.get_dummies(y_test).values[:, 1]
                
                # Class distribution after oversampling
                class_dist_after = data_upsampled['Category'].value_counts()

                # ✅ Dimensi data sebagai tabel Streamlit
                shape_df = pd.DataFrame({
                    'Set': ['X_train', 'y_train', 'X_test', 'y_test'],
                    'Shape': [
                        str(X_train_padded.shape),
                        str(y_train_encoded.shape),
                        str(X_test_padded.shape),
                        str(y_test_encoded.shape)
                    ]
                })
                st.markdown("**Dimensi Dataset Setelah Upsampling:**")
                st.table(shape_df)
                
                st.header("Handling Imbalance Class", divider='blue')
                st.subheader("Class Distribution Before vs After Upsampling")
                fig = plot_class_distribution(
                    y_train, data_upsampled['Category'],
                    "Class Distribution Before Upsampling",
                    "Class Distribution After Upsampling"
                )
                st.pyplot(fig)
                
                
                st.header("Visual Model Summary", divider='blue')
                input_size = np.max(X_train_padded) + 1
                embed_dim = 64
                
                model = build_lstm_model(input_size, embed_dim, max_length)
                
                # Show model architecture
                col1, col2 = st.columns((2))
                with col1:
                    image = Image.open(r"C:\Users\KENT LEE\Documents\TA KENT 2025\views\model_structure.png")
                    st.image(image, caption='Model Structure', use_container_width=True)
                
                st.header("Evaluation & Testing", divider='blue')
                st.subheader("Epoch Time Series")
                # Class weights for imbalanced data
                bias = data_minority.shape[0] / data_majority.shape[0]
                class_weights = {0: 1, 1: 1.6/bias}
                
                # Train model
                history = model.fit(
                    X_train_padded, y_train_encoded,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test_padded, y_test_encoded),
                    class_weight=class_weights
                )
                
                fig = plot_lstm_performance(history)
                st.pyplot(fig)
                
                st.subheader("Parameter Comparison for LSTM")
                results = evaluate_lstm_model(model, X_test_padded, y_test_encoded, y)
                
                # Plot metrics
                metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
                values = [
                    results['accuracy'],
                    results['f1_score'],
                    results['recall'],
                    results['precision']
                ]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(metrics, values, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}',
                            ha='center', va='bottom', fontsize=12)
                
                ax.set_xlabel("Metrics", fontweight='bold')
                ax.set_ylabel("Score", fontweight='bold')
                ax.set_title("Model Performance After Weighting", fontweight='bold')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(results['confusion_matrix'], annot=True, fmt="d", cmap='Blues',
                            xticklabels=['Real User', 'Buzzer'], 
                            yticklabels=['Real User', 'Buzzer'], ax=ax)
                ax.set_title('Confusion Matrix', fontweight='bold')
                ax.set_xlabel('Predicted', fontweight='bold')
                ax.set_ylabel('Actual', fontweight='bold')
                st.pyplot(fig)
                
                st.subheader("Classification Report")
                # Konversi label unik menjadi string
                target_names = [str(label) for label in np.unique(y)]

                # Convert report to DataFrame
                report_df = pd.DataFrame(results['classification_report']).transpose()
                report_df = report_df.drop(columns=["support"], errors="ignore")
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(report_df, annot=True, cmap="coolwarm", fmt=".2f", 
                            linewidths=0.5, ax=ax)
                ax.set_title("Classification Report", fontsize=14, fontweight='bold')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                st.pyplot(fig)
                
                st.subheader("Actual vs LSTM Prediction")
                # Convert predictions to labels
                y_pred = model.predict(X_test_padded)
                y_pred_classes = (y_pred > 0.5).astype(int)
                y_pred_labels = ['Buzzer' if pred == 1 else 'Real User' for pred in y_pred_classes]
                
                dfTest = pd.DataFrame({
                    "Tweet": X_test,
                    "Category (Data Test)": y_test,
                    "LSTM Prediction": y_pred_labels
                })
                st.write(dfTest.head(20))

main()
