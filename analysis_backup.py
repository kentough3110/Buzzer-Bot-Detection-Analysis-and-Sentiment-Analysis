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
import seaborn as sns
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
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
nltk.download('stopwords')
stop_words = stopwords.words('indonesian')
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # type: ignore
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
warnings.filterwarnings('ignore')

st.title(" :bar_chart: Data Analysis")
st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>',unsafe_allow_html=True)

counter = 0
fl = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_excel(filename)
    counter += 1
else:
    os.chdir(r"C:\Users\KENT LEE\Documents\TA KENT 2025")
    df = pd.read_excel("doktifall.xlsx")
    
if counter == 0:
    # ---------------------
    # ---= RAW DATASET =---
    # ---------------------
    st.header("Raw Tweet Dataset", anchor=False, divider="blue")    

    # Memisahkan tanggal dan waktu untuk setiap baris dalam DataFrame
    for index, row in df.iterrows():
        try:
            date_time_obj = datetime.strptime(row['created_at'], '%a %b %d %H:%M:%S %z %Y')
            df.at[index, 'tanggal'] = date_time_obj.strftime('%a %b %d')
            df.at[index, 'waktu'] = date_time_obj.strftime('%H:%M:%S')
        except ValueError:
            st.write(f"Done: data berhasil di bagi: {row['created_at']}")

    # Mengambil feature yang dibutuhkan
    df = pd.DataFrame(df[['tanggal','waktu','username','favorite_count','quote_count','reply_count','retweet_count','full_text']])
    st.write(df.head(5))
    # --------------------------------
    # --= EDA BEFORE PREPROCESSING =--
    # --------------------------------
    st.header("EDA Before Preprocessing", divider="blue")
    col1, col2 = st.columns((2))
    with col1:
        st.subheader("Words Frequency Distribution")
        # Process Text
        text = " ".join(df["full_text"])
        tokens = text.split()
        word_counts = Counter(tokens)

        # Sort untuk Top 10 kata
        top_words = word_counts.most_common(10)
        word, count = zip(*top_words)

        # Color Palette
        colors = plt.cm.tab10(range(len(word)))

        # Buat Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(word, count, color=colors)
        ax.set_xlabel("Kata yang Sering Muncul", fontsize=12, fontweight='bold')
        ax.set_ylabel("Jumlah Kata", fontsize=12, fontweight='bold')
        ax.set_title("Frekuensi Kata", fontsize=18, fontweight='bold')
        ax.set_xticks(range(len(word)))
        ax.set_xticklabels(word, rotation=45)

        # Tambah label 
        for bar, num in zip(bars, count):
            ax.text(bar.get_x() + bar.get_width() / 2, num + 0.5, str(num), fontsize=12, color='black', ha='center')

        st.pyplot(fig)
        
    with col2:
        st.subheader("Wordcloud Dataset")
        # Mengisi nilai null dengan blank string ('')
        df['full_text'] = df['full_text'].fillna('')

        # Merged text dari feature 'full_text'
        text = ' '.join(df['full_text'].astype(str).tolist())

        stopwords = set(STOPWORDS)
        stopwords.update(['https', 'co', 'RT', '...', 'amp'])

        wc = WordCloud(stopwords=stopwords, background_color="white", max_words=500, width=800, height=450)
        wc.generate(text)

        # Buat figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    
    # --------------------
    # ----= CLEANING =----
    # --------------------
    st.header("Preprocessing", divider="blue")
    st.subheader("Data Cleaning")
    # Function untuk cleaning text
    def clean_text(text):
        if not isinstance(text, str):
            return text
        # Menghapus atau replce text sesuai pola regex
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Hapus URL
        text = re.sub(r'@[^\s]+', '', text)  # Hapus username
        text = re.sub(r'<.*?>', '', text)  # Hapus HTML tags
        text = re.sub(r'['
            u'\U0001F600-\U0001F64F'  # Emoticons
            u'\U0001F300-\U0001F5FF'  # Symbols & pictographs
            u'\U0001F680-\U0001F6FF'  # Transport & map symbols
            u'\U0001F700-\U0001F77F'  # Alchemical symbols
            u'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
            u'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
            u'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
            u'\U0001FA00-\U0001FA6F'  # Chess Symbols
            u'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
            u'\U0001F004-\U0001F0CF'  # Additional emoticonshn 
            u'\U0001F1E0-\U0001F1FF'  # Flags
            ']+', '', text, flags=re.UNICODE)  # Hapus emoji
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Hapus simbol
        text = re.sub(r'\d', '', text)  # Hapus angka

        return text
    df = pd.DataFrame(df['full_text'])
    # Apply function cleaning ke feature 'full_text'
    df['cleaning'] = df['full_text'].apply(clean_text)

    st.write(df.head(5))

    # --------------------
    # --= NLP PIPELINE =--
    # --------------------
    st.header("NLP Pipeline", divider="blue")

    # CASE FOLDING
    st.subheader("Case Folding")
    def caseFolding(text):
        if isinstance(text, str):
            lowercase_text = text.lower()
            return lowercase_text
        else:
            return text

    df['case_folding'] = df['cleaning'].apply(caseFolding)

    df =  pd.DataFrame(df[['cleaning','case_folding']])
    st.write(df.head(5))

    # CASE NORMALIZATION
    st.subheader("Normalization")  
    # Import kamus kata tidak baku
    kamus_data = pd.read_excel(r"C:\Users\KENT LEE\Documents\TA KENT 2025\lexicon\kamuskatabaku.xlsx")
    kamus_tidak_baku = dict(zip(kamus_data['kata_tidak_baku'], kamus_data['kata_baku']))  

    # Function untuk replace kata non baku
    def replace_taboo_words(text, kamus_tidak_baku):
        if not isinstance(text, str):
            return '', [], [], []

        replaced_words, kalimat_baku, kata_diganti, kata_tidak_baku = [], [], [], []

        for word in text.split():
            if word in kamus_tidak_baku:
                baku_word = kamus_tidak_baku[word]
                if isinstance(baku_word, str) and baku_word.isalpha():
                    replaced_words.append(baku_word)
                    kalimat_baku.append(baku_word)
                    kata_diganti.append(word)
                    kata_tidak_baku.append(hash(word))
            else:
                replaced_words.append(word)

        return ' '.join(replaced_words), kalimat_baku, kata_diganti, kata_tidak_baku
    
    # Implement function replace kata tidak baku
    df['normalization'], df['Kata_Baku'], df['Kata_Tidak_Baku'], df['Kata_Tidak_Baku'] = zip(*df['case_folding'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku)))

    df =  pd.DataFrame(df[['case_folding','normalization']])
    st.write(df.head(5))

    # TOKENIZATION
    st.subheader("Tokenization")  

    df['tokenization'] = df['normalization'].str.split()

    df =  pd.DataFrame(df[['normalization','tokenization']])
    st.write(df.head(5))

    # STOPWORDS
    st.subheader("Stopwords Removal")  

    def remove_stopwords(text):
        return [word for word in text if word not in stop_words]

    df['stopword_removal'] = df['tokenization'].apply(lambda x: remove_stopwords(x))

    df =  pd.DataFrame(df[['tokenization','stopword_removal']])
    st.write(df.head(5))

    # STEMMING
    st.subheader("Stemming")  

    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()

    # def stem_text(text):
    #     return [stemmer.stem(word) for word in text]

    # df['stemming_data'] = df['stopword_removal'].apply(lambda x: ' '.join(stem_text(x)))
    # df = pd.read_csv(r"C:\Users\KENT LEE\Documents\TA KENT 2025\doktif_after_preprocessing.csv")

    # df =  pd.DataFrame(df[['stopword_removal','stemming_data']])
    # st.write(df.head(5))

    # ENGLISH TRANSLATION
    st.subheader("English Translation")  

    df = pd.read_csv(r"C:\Users\KENT LEE\Documents\TA KENT 2025\doktif_after_preprocessing.csv", encoding="ISO-8859-1")

    df =  pd.DataFrame(df[['stemming_data','eng_translation']])
    st.write(df)

    # -------------------------------
    # --= EDA AFTER PREPROCESSING =--
    # -------------------------------
    st.header("EDA After Preprocessing", divider="blue")
    col1, col2 = st.columns((2))
    with col1:
        st.subheader("Words Frequency Distribution")
        # Process Text
        df['stemming_data'] = df['stemming_data'].astype(str).fillna('')
        text = " ".join(df['stemming_data'])

        # Define Stopwords
        stopwords = set(STOPWORDS)
        stopwords.update(['https', 'co', 'RT', '...', 'amp', 'ya', 'lu', 'gue', 'sih', 'sama', 'nih', 'tuh', 'bikin', 'kak', 'dapat', 'kayak'])

        # Tokenize & Filter Stopwords
        tokens = [word for word in text.split() if word.lower() not in stopwords]
        word_counts = Counter(tokens)

        # Get Top 10 Words
        top_words = word_counts.most_common(10)
        if not top_words:
            st.warning("Tidak ada kata yang cukup signifikan untuk ditampilkan.")
        else:
            word, count = zip(*top_words)

        # Define Color Palette
        colors = plt.cm.tab10(range(len(word)))

        # Create Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size for better alignment in Streamlit
        bars = ax.bar(word, count, color=colors)
        ax.set_xlabel("Kata yang Sering Muncul", fontsize=10, fontweight='bold')
        ax.set_ylabel("Jumlah Kata", fontsize=10, fontweight='bold')
        ax.set_title("Frekuensi Kata", fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(word)))
        ax.set_xticklabels(word, rotation=45)

        # Add Labels Above Bars
        for bar, num in zip(bars, count):
            ax.text(bar.get_x() + bar.get_width() / 2, num + 0.5, str(num), fontsize=10, color='black', ha='center')

        st.pyplot(fig)
        
    with col2:
        st.subheader("Wordcloud Dataset")
        # Mengisi nilai null dengan blank string ('')
        df['stemming_data'] = df['stemming_data'].astype(str).fillna('')

        # Merged text dari feature 'stemming_data'
        text = ' '.join(df['stemming_data'].astype(str).tolist())

        # Define Stopwords
        stopwords = set(STOPWORDS)
        stopwords.update(['https', 'co', 'RT', '...', 'amp', 'ya', 'lu', 'gue', 'sih', 'sama', 'nih', 'tuh', 'bikin', 'kak', 'dapat', 'kayak'])

        # Generate WordCloud
        wc = WordCloud(stopwords=stopwords, background_color="white", max_words=500, width=800, height=450)
        wc.generate(text)

        # Create Figure for Streamlit
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")  # Hide axes

        st.pyplot(fig)

    # -----------------------------------
    # --= DATASET AFTER PREPROCESSING =--
    # -----------------------------------
    st.header("Dataset After Preprocessing", divider="blue")

    df = pd.read_csv(r"C:\Users\KENT LEE\Documents\TA KENT 2025\doktif_after_preprocessing.csv", encoding="ISO-8859-1")
    # Mengambil feature yang dibutuhkan
    df = pd.DataFrame(df[['tanggal','waktu','username','stemming_data', 'eng_translation']])

    st.write(df.head(5)) 

    # ---------------------
    # --= INSET LEXICON =--
    # ---------------------
    st.header("InSet Lexicon", divider = 'blue') 
    st.subheader("Labelling") 
    # Baca file leksikon full dari CSV
    leksikon_df = pd.read_csv(r"C:\Users\KENT LEE\Documents\TA KENT 2025\lexicon\lexiconfull.csv")

    # Pisahkan kata-kata positif dan negatif berdasarkan nilai di kolom 'weight'
    lexicon_positive = {row['word']: int(row['weight']) for _, row in leksikon_df[leksikon_df['weight'] > 0].iterrows()}
    lexicon_negative = {row['word']: int(row['weight']) for _, row in leksikon_df[leksikon_df['weight'] < 0].iterrows()}

    # Fungsi untuk menentukan sentimen dan menghitung skornya
    def lexicon_polarity_indonesia(text):
        # Inisialisasi skor dan penyimpanan kata
        score = 0
        word_weights = {}
        positive_words = []
        negative_words = []

        if isinstance(text, str):
            for word in text.split():
                if word in lexicon_positive:
                    score += lexicon_positive[word]
                    word_weights[word] = lexicon_positive[word]
                    positive_words.append(word)
                elif word in lexicon_negative:
                    score += lexicon_negative[word]
                    word_weights[word] = lexicon_negative[word]
                    negative_words.append(word)
                else:
                    word_weights[word] = 0  # Kata netral

        # Tentukan label sentimen
        if score > 0:
            polarity = "Positif"
        elif score < 0:
            polarity = "Negatif"
        else:
            polarity = "Netral"

        return score, polarity

    # Terapkan fungsi ke dataset
    df[['Score', 'Sentiment']] = df['stemming_data'].apply(lambda x: pd.Series(lexicon_polarity_indonesia(x)))

    # Kategorikan sebagai Buzzer atau Real User
    df['Category'] = df['Sentiment'].apply(lambda x: "Buzzer" if x in ["Positif", "Negatif"] else "Real User")

    # Mengambil feature yang dibutuhkan
    df = pd.DataFrame(df[['stemming_data', 'Score', 'Sentiment', 'Category']])

    st.write(df.head(10))

    col1, col2 = st.columns((2))
    # AFTER PREPROCESSING
    with col1:
        st.subheader("Sentiment Analysis Distribution")
        # Drop NaN values in Sentiment column
        df = df.dropna(subset=["Sentiment"])

        # Count sentiment occurrences
        sentiment_count = df['Sentiment'].value_counts()

        # Seaborn Style
        sns.set_style('whitegrid')

        # Create Figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create Bar Plot
        sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette='tab10', ax=ax)

        # Set Title & Labels
        ax.set_title('Jumlah Analisis Sentimen Metode InSet Lexicon', fontsize=14, pad=20, fontweight='bold')
        ax.set_xlabel('Class Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count Tweet', fontsize=12, fontweight='bold')

        # Total Sentiment Count
        total = sentiment_count.sum()

        # Tambah Labels on Bars
        for i, count in enumerate(sentiment_count.values):
            percentage = f"{100 * count / total:.2f}%"
            ax.text(i, count + 0.10, f"{count}\n({percentage})", ha='center', va='bottom')

        st.pyplot(fig)

    # BUZZER DETECTION
    with col2:
        st.subheader("Buzzer vs Real User Distribution")
        df = df.dropna(subset=["Category"])

        # Count occurrences of each category
        buzzer_counts = df['Category'].value_counts()

        # Compute percentages
        total = buzzer_counts.sum()
        percentages = (buzzer_counts / total) * 100

        # Create Figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create Bar Plot
        bars = ax.bar(buzzer_counts.index, buzzer_counts.values, color=['tab:blue', 'tab:orange'])

        # Add Labels Above Bars
        for bar, count, percentage in zip(bars, buzzer_counts.values, percentages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{count}\n({percentage:.1f}%)", ha='center', va='bottom', fontsize=12)

        # Set Labels & Title
        ax.set_xlabel("Category", fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        ax.set_title("Jumlah Buzzer vs Real User Metode InSet Lexicon", fontsize=14, pad=20, fontweight='bold')

        # Display in Streamlit
        st.pyplot(fig)

    # Load the preprocessed dataset
    df1 = pd.read_csv(r"C:\Users\KENT LEE\Documents\TA KENT 2025\doktif_after_preprocessing.csv", encoding="ISO-8859-1")

    # Create df with selected columns
    df = df[['Score', 'Sentiment', 'Category']].copy()

    # Ensure df1['eng_translation'] has the same length or realign indices
    df['eng_translation'] = df1['eng_translation'].values

    # ---------------------
    # --= VADER LEXICON =--
    # ---------------------
    st.header("VADER Lexicon", divider = 'blue')
    st.subheader("Labelling")
    # Unduh lexicon VADER jika belum terinstal
    nltk.download('vader_lexicon')
    # Inisialisasi SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # Fungsi untuk menentukan sentimen menggunakan VADER
    def determine_vader_sentiment(text):
        if isinstance(text, str):
            sentiment_scores = sia.polarity_scores(text)  # Ambil semua skor
            compound_score = sentiment_scores['compound']

            # Tentukan label sentimen berdasarkan nilai skor
            if compound_score >= 0.05:
                sentiment = "Positif"
            elif compound_score <= -0.05:
                sentiment = "Negatif"
            else:
                sentiment = "Netral"

            return sentiment_scores['pos'], sentiment_scores['neg'], sentiment_scores['neu'], compound_score, sentiment

        return 0, 0, 0, 0, "Netral"

    # Hapus kolom 'Score' dan 'Sentiment' untuk menghindari duplikasi
    df.drop(columns=['Score', 'Sentiment', 'Category'], inplace=True)

    # Terapkan fungsi ke kolom 'eng_translation'
    df[['Positive', 'Negative', 'Neutral', 'Compound Score', 'Sentiment']] = df['eng_translation'].apply(lambda x: pd.Series(determine_vader_sentiment(x)))

    # Kategorikan sebagai Buzzer atau Real User
    df['Category'] = df['Sentiment'].apply(lambda x: "Buzzer" if x in ["Positif", "Negatif"] else "Real User")

    # Tampilkan 10 baris pertama hasil analisis sentimen
    st.write(df.head(10))

    col1,col2 = st.columns((2))
    with col1:
        st.subheader("Sentiment Analysis Distribution")
        # Hitung jumlah masing-masing sentimen
        sentiment_count = df['Sentiment'].value_counts()

        # Atur gaya seaborn
        sns.set_style('whitegrid')

        # Buat figure dan axis untuk plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Buat barplot untuk visualisasi jumlah sentimen
        sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette='tab10', ax=ax)

        # Tambahkan judul dan label sumbu
        ax.set_title('Jumlah Analisis Sentimen Metode VADER Sentiment', fontsize=14, pad=20, fontweight='bold')
        ax.set_xlabel('Class Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tweet Count', fontsize=12, fontweight='bold')

        # Hitung total jumlah data
        total = len(df['Sentiment'])

        # Tambah label jumlah dan persentase di atas batang
        for i, count in enumerate(sentiment_count.values):
            percentage = f"{100 * count / total:.2f}%"
            ax.text(i, count + 0.10, f"{count}\n({percentage})", ha='center', va='bottom')

        st.pyplot(fig)

    with col2:
        st.subheader("Buzzer vs Real User Distribution")
        # Menghitung jumlah masing-masing kategori
        buzzer_counts = df['Category'].value_counts()

        # Menghitung persentase
        total = buzzer_counts.sum()
        percentages = (buzzer_counts / total) * 100

        # Membuat figure dan axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Membuat bar chart
        bars = ax.bar(buzzer_counts.index, buzzer_counts.values, color=['tab:blue', 'tab:orange'])

        # Menambahkan label count dan persentase di atas batang
        for bar, count, percentage in zip(bars, buzzer_counts.values, percentages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{count}\n({percentage:.1f}%)", ha='center', va='bottom', fontsize=12)

        # Menambahkan label sumbu dan judul
        ax.set_xlabel("Category", fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        ax.set_title("Jumlah Buzzer vs Real User Metode VADER Lexicon", fontsize=14, pad=15, fontweight='bold')

        st.pyplot(fig)

    with st.form("settings_form"):
        option = st.selectbox(
            "Choose Lexicon Methodology:",
            ("InSet Lexicon", "VADER Lexicon"),
            index=None,
            placeholder="Select lexicon method..."
        )

        option_algo = st.selectbox(
            "Choose Algorithm:",
            ("Machine Learning", "Deep Learning"),
            index=None,
            placeholder="Select algorithm..."
        )

        run_btn = st.form_submit_button("Run", type="primary")

    df = pd.read_csv(r"C:\Users\KENT LEE\Documents\TA KENT 2025\doktif_labelling_inset.csv", encoding="ISO-8859-1")

    if option == "InSet Lexicon" and run_btn:
        df = df.dropna(subset=['stemming_data'])
        x = df['stemming_data']
        y = df['Category']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        st.header("Train Test Split (80:20)", divider='blue')
        # Hitung jumlah data training dan testing
        dtrain = len(X_train)
        dtest = len(X_test)
        col1, col2 = st.columns((2))
        with col1:
            # Buat figure dan axis
            fig, ax = plt.subplots(figsize=(8, 6))

            # Buat bar chart
            bars = ax.bar(['Data Training', 'Data Testing'], [dtrain, dtest], color=['tab:blue', 'tab:orange'])

            # Tambahkan label jumlah dan persentase di atas batang
            for bar, count in zip(bars, [dtrain, dtest]):
                percentage = f"{count / (dtrain + dtest) * 100:.2f}%"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, 
                        f"{count}\n({percentage})", ha='center', va='bottom', fontsize=12)

            # Tambahkan label sumbu dan judul
            ax.set_xlabel("Kategori Data", fontsize=12, fontweight='bold')
            ax.set_ylabel("Jumlah Data", fontsize=12, fontweight='bold')
            ax.set_title("Jumlah Data Training dan Data Testing", fontsize=14, pad=15, fontweight='bold')

            st.pyplot(fig)

        st.header("Data Transformation", divider='blue')
        if option_algo == "Machine Learning":
            st.subheader("TF-IDF Vectorizing")

            tfidf = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, ngram_range=(1,1), stop_words=stop_words)

            x_train_tfidf = tfidf.fit_transform(X_train)
            x_test_tfidf = tfidf.transform(X_test)

            tfidf_data = tfidf.fit_transform(df['stemming_data'])
            tfidf_df = pd.DataFrame(tfidf_data.toarray(), columns=tfidf.get_feature_names_out())

            top_n = 10

            top_tfidf_words = tfidf_df.apply(lambda x: x.nlargest(top_n).index.tolist(), axis=1)
            top_tfidf_values = tfidf_df.apply(lambda x: x.nlargest(top_n).values.tolist(), axis=1)

            # Buat dataframe
            top_tfidf_df = pd.DataFrame({
                "Top Words": top_tfidf_words,
                "Top Values": top_tfidf_values
            })

            st.write(top_tfidf_df.head(10))
        elif option_algo == "Deep Learning":
            st.subheader("Sequences Conversion and Padding")
            # Pisahkan majority dan minority class
            data_majority = df[df['Category'] == 'Buzzer']
            data_minority = df[df['Category'] == 'Real User']

            # Hitung rasio bias (untuk penalty tiap misklasifikasi)
            bias = data_minority.shape[0] / data_majority.shape[0]

            # Merged kembali X_train dan y_train ke dalam 1 df untuk oversampling
            train = pd.DataFrame({'eng_translation': X_train, 'Category': y_train})

            # Merged kembali X_test dan y_test ke dalam 1 df untuk pengecekan distribusi kelas
            test = pd.DataFrame({'eng_translation': X_test, 'Category': y_test})

            # ✅ Distribusi kelas sebagai tabel Streamlit
            class_distribution_df = pd.DataFrame({
                'Set': ['Training', 'Training', 'Testing', 'Testing'],
                'Category': ['Buzzer', 'Real User', 'Buzzer', 'Real User'],
                'Count': [
                    (train['Category'] == 'Buzzer').sum(),
                    (train['Category'] == 'Real User').sum(),
                    (test['Category'] == 'Buzzer').sum(),
                    (test['Category'] == 'Real User').sum()
                ]
            })
            st.markdown("**Distribusi Kelas (Data Train & Data Test):**")
            st.table(class_distribution_df)

            # Distribusi kelas sebelum oversampling
            class_distribution_before = y_train.value_counts()

            # Pisahkan kembali majority dan minority class untuk oversampling
            data_majority = train[train['Category'] == 'Buzzer']
            data_minority = train[train['Category'] == 'Real User']

            # Apply oversample untuk minority class
            data_minority_upsampled = resample(
                data_minority,
                replace=True,
                n_samples=data_majority.shape[0],
                random_state=123
            )

            # Gabungkan kembali dan acak
            data_upsampled = pd.concat([data_majority, data_minority_upsampled])
            data_upsampled = shuffle(data_upsampled, random_state=42)

            # Tokenisasi
            vocab_size = 5000
            tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
            tokenizer.fit_on_texts(df['eng_translation'].values)

            # Train set
            # Konversi teks ke dalam urutan angka (sequences)
            X_train_sequences = tokenizer.texts_to_sequences(data_upsampled['eng_translation'].values)
            max_length = max([len(seq) for seq in X_train_sequences])
            # Padding sequences agar memiliki panjang yang sama
            X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length)
            y_train_encoded = pd.get_dummies(data_upsampled['Category']).values[:, 1]

            # Test set
            # Konversi teks ke dalam urutan angka (sequences)
            X_test_sequences = tokenizer.texts_to_sequences(X_test.values)
            # Padding sequences for equalizing dimension (length)
            X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length)
            # One-hot encode labels array 1D
            y_test_encoded = pd.get_dummies(y_test).values[:, 1]

            # Menghitung distribusi kelas setelah oversampling
            class_distribution_after = data_upsampled['Category'].value_counts()

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
            st.markdown("**Dimensi Dataset Setelah Oversampling:**")
            st.table(shape_df)

        # HANDLING IMBALANCE
        st.header("Handling Imbalance Class", divider='blue')
        if option_algo == "Machine Learning":
            st.subheader("Class Distribution Before vs After SMOTE")
            
            # Menghitung distribusi kelas sebelum menerapkan SMOTE
            class_distribution_before = y_train.value_counts()

            # Menerapkan teknik oversampling SMOTE untuk menyeimbangkan data
            smote = SMOTE(random_state=42)
            x_train_tfidf_resampled, y_train_resampled = smote.fit_resample(x_train_tfidf, y_train)

            # Menghitung distribusi kelas setelah penerapan SMOTE
            class_distribution_after = pd.Series(y_train_resampled).value_counts()
        
            # Membuat figure untuk visualisasi dist ribusi kelas
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Fungsi untuk menambahkan label angka di atas batang grafik
            def add_labels(ax, values):
                for i, v in enumerate(values):
                    ax.text(i, v + 10, str(v), ha='center', fontsize=12, color='black')

            # Grafik sebelum menerapkan SMOTE
            axes[0].bar(class_distribution_before.index.astype(str), class_distribution_before.values, color='tab:blue')
            axes[0].set_title("Distribusi Kelas Sebelum SMOTE", fontweight='bold')
            axes[0].set_xlabel("Kategori", fontweight='bold')
            axes[0].set_ylabel("Jumlah", fontweight='bold')
            add_labels(axes[0], class_distribution_before.values)

            # Grafik setelah menerapkan SMOTE
            axes[1].bar(class_distribution_after.index.astype(str), class_distribution_after.values, color='tab:orange')
            axes[1].set_title("Distribusi Kelas Setelah SMOTE", fontweight='bold')
            axes[1].set_xlabel("Kategori", fontweight='bold')
            axes[1].set_ylabel("Jumlah", fontweight='bold')
            add_labels(axes[1], class_distribution_after.values)

            plt.tight_layout()

            st.pyplot(fig)
        elif option_algo == "Deep Learning":
            st.subheader("Class Distribution Before vs After Oversampling")
            # Buat figure dan axes
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Fungsi untuk menambahkan label ke atas bar
            def add_labels(ax, values):
                for i, v in enumerate(values):
                    ax.text(i, v + 10, str(v), ha='center', fontsize=12, color='black')

            # Plot distribusi sebelum oversampling
            axes[0].bar(class_distribution_before.index.astype(str), class_distribution_before.values, color='tab:blue')
            axes[0].set_title("Class Distribution Before Oversampling", fontweight='bold')
            axes[0].set_xlabel("Kategori", fontweight='bold')
            axes[0].set_ylabel("Jumlah", fontweight='bold')
            add_labels(axes[0], class_distribution_before.values)

            # Plot distribusi setelah oversampling
            axes[1].bar(class_distribution_after.index.astype(str), class_distribution_after.values, color='tab:orange')
            axes[1].set_title("Class Distribution After Oversampling", fontweight='bold')
            axes[1].set_xlabel("Kategori", fontweight='bold')
            axes[1].set_ylabel("Jumlah", fontweight='bold')
            add_labels(axes[1], class_distribution_after.values)

            # Tata letak dan tampilkan ke Streamlit
            plt.tight_layout()
            st.pyplot(fig)    

        if option_algo == "Machine Learning":
            # Modeling before smote
            # Inisialisasi model
            models = {
                "Support Vector Machine": SVC(kernel='linear', random_state=42),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting Machine (GBM)": GradientBoostingClassifier(random_state=42),
            }

            # Dictionary untuk menyimpan hasil evaluasi model sebelum SMOTE
            before_smt_results = {}

            # Looping melalui setiap model untuk pelatihan dan evaluasi
            for model_name, model in models.items():
                # Konversi sparse matrix ke dense array jika diperlukan
                x_train_dense = x_train_tfidf.toarray() if hasattr(x_train_tfidf, "toarray") else x_train_tfidf
                x_test_dense = x_test_tfidf.toarray() if hasattr(x_test_tfidf, "toarray") else x_test_tfidf

                # Mengukur waktu pelatihan
                start_time = time.time()
                model.fit(x_train_dense, y_train)
                end_time = time.time()
                training_time = end_time - start_time
                minutes, seconds = divmod(int(training_time), 60)
                formatted_time = f"{minutes:02d} m {seconds:02d} s"

                # Melakukan prediksi
                y_pred = model.predict(x_test_dense)

                # Menyimpan hasil evaluasi
                before_smt_results[model_name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                    "training_time": formatted_time
                }

            # Modeling after smote
            # Inisialisasi model
            models = {
                "Support Vector Machine": SVC(kernel='linear', random_state=42),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting Machine (GBM)": GradientBoostingClassifier(random_state=42),
            }

            # Dictionary untuk menyimpan hasil evaluasi setelah SMOTE
            after_smt_results = {}
            predictions = {}

            # Looping melalui setiap model untuk pelatihan dan evaluasi
            for model_name, model in models.items():
                # Konversi sparse matrix ke dense array setelah SMOTE jika diperlukan
                X_train_dense = x_train_tfidf_resampled.toarray() if hasattr(x_train_tfidf_resampled, "toarray") else x_train_tfidf_resampled
                X_test_dense = x_test_tfidf.toarray() if hasattr(x_test_tfidf, "toarray") else x_test_tfidf

                # Mengukur waktu pelatihan
                start_time = time.time()

                # Melatih model
                model.fit(X_train_dense, y_train_resampled)

                # Melakukan prediksi
                y_pred = model.predict(X_test_dense)

                end_time = time.time()
                training_time = end_time - start_time
                minutes, seconds = divmod(int(training_time), 60)
                formatted_time = f"{minutes:02d} m {seconds:02d} s"

                # Menyimpan hasil prediksi
                predictions[model_name] = y_pred

                # Menyimpan hasil evaluasi model
                after_smt_results[model_name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                    "training_time": formatted_time
                }
        elif option_algo == "Deep Learning":
            st.header("Visual Model Summary", divider='blue')
            # For Embedding (Hyperparameter to tune) 🔍
            input_size = np.max(X_train_padded) + 1
            # Try power of 2 (16,32,64,128,...,etc) or 50,100,200,300
            embed_dim = 64
            # Padding sequences for equalizing dimension (length)
            max_length = max([len(seq) for seq in X_train_sequences])

            # Define a model
            model = Sequential()
            # Add an embedding layer with hyperparameter based on optimal value
            model.add(Embedding(input_dim=input_size, output_dim=embed_dim, input_length=max_length))
            # Build model explicitly
            model.build(input_shape=(None, max_length))
            # Add a LSTM layer
            model.add(LSTM(64, dropout=0.2))
            # model.add(Dropout(0.2))
            # Add a dense layer with ReLU activation
            model.add(Dense(32, activation='relu'))
            # Output layer with sigmoid activation for binary crossentropy loss function
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
            col1, col2 = st.columns((2))
            # # Visualize model structure
            # plot_model(model, show_shapes=True, to_file="model_structure.png")
            # --- A. Tampilkan struktur model dalam bentuk gambar ---
            with col1:
                image = Image.open(r"C:\Users\KENT LEE\Documents\TA KENT 2025\views\model_structure.png")
                st.image(image, caption='Model Structure', use_container_width=True)

            # # --- B. Tampilkan tabel layer breakdown ---
            # layer_info = []
            # for layer in model.layers:
            #     layer_info.append({
            #         "Layer Type": layer.__class__.__name__,
            #         "Output Shape": str(layer.output_shape),
            #         "Param #": layer.count_params()
            #     })

            # df_layers = pd.DataFrame(layer_info)
            # st.markdown("Layer Detail Table")
            # st.table(df_layers)

        # LSTM ONLY: EPOCH TIME SERIES    
        if option_algo == "Deep Learning":
            st.header("Epoch Time Series", divider='blue')
            # Adding weights untuk penalty dalam setiap missclassification pada sesi training
            class_weights = {0:1, 1:1.6/bias}
            # Training size (Try smaller power of 2 such as 32,64,128)
            batch_size = 32
            # Model Train
            time_series_weight = model.fit(X_train_padded, y_train_encoded, epochs=10, batch_size=batch_size, validation_data=(X_test_padded, y_test_encoded), 
                class_weight=class_weights)
            
            # Buat figure dengan dua subplot dalam satu baris
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot Accuracy
            axes[0].plot(time_series_weight.history['accuracy'], marker='o')
            axes[0].plot(time_series_weight.history['val_accuracy'], marker='s')
            axes[0].set_title('Model Accuracy', fontweight='bold')
            axes[0].set_xlabel('Epoch', fontweight='bold')
            axes[0].set_ylabel('Accuracy', fontweight='bold')
            axes[0].set_xticks(range(1, len(time_series_weight.history['accuracy']) + 1))
            # Legend di bawah plot
            axes[0].legend(['Train', 'Validation'],
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.15),
                        ncol=2)

            # Plot Loss
            axes[1].plot(time_series_weight.history['loss'], marker='o')
            axes[1].plot(time_series_weight.history['val_loss'], marker='s')
            axes[1].set_title('Model Loss', fontweight='bold')
            axes[1].set_xlabel('Epoch', fontweight='bold')
            axes[1].set_ylabel('Loss', fontweight='bold')
            axes[1].set_xticks(range(1, len(time_series_weight.history['loss']) + 1))
            # Legend di bawah plot
            axes[1].legend(['Train', 'Validation'],
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.15),
                        ncol=2)

            # Adjust layout & tampilkan di Streamlit
            plt.tight_layout(rect=[0, 0.1, 1, 1])  # Sisakan ruang di bawah untuk legends
            st.pyplot(fig)
            

        if option_algo == "Machine Learning":
            #ACCURACY X F1-SCORE COMPARATION VISUALIZATION
            st.header("Parameter Comparison for Model", divider='blue')
            col1, col2 = st.columns((2))
            with col1:
                # Ambil akurasi dari hasil sebelum dan sesudah SMOTE
                before_accuracies = {model: result['accuracy'] for model, result in before_smt_results.items()}
                after_accuracies = {model: result['accuracy'] for model, result in after_smt_results.items()}

                models = list(before_accuracies.keys())  # Pastikan urutan model sama
                before_values = [before_accuracies[model] for model in models]
                after_values = [after_accuracies[model] for model in models]

                # Menentukan posisi bar
                x = np.arange(len(models))  # Label lokasi
                width = 0.35  # Lebar bar

                # Membuat plot dengan Matplotlib
                fig, ax = plt.subplots(figsize=(8, 5))
                bars1 = ax.bar(x - width/2, before_values, width, label='Before SMOTE', color='tab:blue')
                bars2 = ax.bar(x + width/2, after_values, width, label='After SMOTE', color='tab:orange')

                # Menambahkan teks di atas setiap bar
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}',
                                ha='center', va='bottom', fontsize=10)

                # Label dan judul
                ax.set_xlabel("Model", fontweight='bold')
                ax.set_ylabel("Akurasi", fontweight='bold')
                ax.set_title("Perbandingan Akurasi Model Sebelum dan Sesudah SMOTE", fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=15)

                # Menampilkan legend di luar plot
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                plt.ylim(0, 1)  # Batas atas sumbu Y
                plt.tight_layout()

                st.subheader("Accuracy Parameter Comparison Before vs After SMOTE")
                st.pyplot(fig)

            with col2:
                # Ambil F1-Score dari hasil sebelum dan sesudah SMOTE
                before_f1_scores = {model: result['f1_score'] for model, result in before_smt_results.items()}
                after_f1_scores = {model: result['f1_score'] for model, result in after_smt_results.items()}

                models = list(before_f1_scores.keys())  # Pastikan urutan model sama
                before_values = [before_f1_scores[model] for model in models]
                after_values = [after_f1_scores[model] for model in models]

                # Menentukan posisi bar
                x = np.arange(len(models))  # Label lokasi
                width = 0.35  # Lebar bar

                # Membuat plot dengan Matplotlib
                fig, ax = plt.subplots(figsize=(8, 5))
                bars1 = ax.bar(x - width/2, before_values, width, label='Before SMOTE', color='tab:blue')
                bars2 = ax.bar(x + width/2, after_values, width, label='After SMOTE', color='tab:orange')

                # Menambahkan teks di atas setiap bar
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}',
                                ha='center', va='bottom', fontsize=10)

                # Label dan judul
                ax.set_xlabel("Model", fontweight='bold')
                ax.set_ylabel("F1-Score", fontweight='bold')
                ax.set_title("Perbandingan F1-Score Model Sebelum dan Sesudah SMOTE", fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=15)

                # Pindahkan legend ke luar chart
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                plt.ylim(0, 1)  # Batas atas sumbu Y
                plt.tight_layout()

                # **Menampilkan plot di Streamlit**
                st.subheader("F1-Score Parameter Comparison Before vs After SMOTE")
                st.pyplot(fig)
        elif option_algo == "Deep Learning":
            #ACCURACY, F1-SCORE, RECALL, PRECISION VISUALIZATION
            st.header("Parameter Comparison for LSTM", divider='blue')
            # Prediksi
            y_pred = model.predict(X_test_padded)
            y_pred_classes = (y_pred > 0.5).astype(int)

            # Hitung metrik
            accuracy = accuracy_score(y_test_encoded, y_pred_classes)
            recall = recall_score(y_test_encoded, y_pred_classes)
            precision = precision_score(y_test_encoded, y_pred_classes)
            f1 = f1_score(y_test_encoded, y_pred_classes)

            # Siapkan data untuk plot
            metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
            values = [accuracy, f1, recall, precision]

            # Buat plot bar
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(metrics, values, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])

            # Tambahkan label nilai di atas bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}',
                        ha='center', va='bottom', fontsize=12)

            # Label dan judul
            ax.set_xlabel("Metrics", fontweight='bold')
            ax.set_ylabel("Score", fontweight='bold')
            ax.set_title("Model Performance After Weighting", fontweight='bold')
            ax.set_ylim(0, 1)

            # Tampilkan di Streamlit
            st.pyplot(fig)

        #CONFUSION MATRIX
        st.header("Evaluation & Testing", divider='blue')
        if option_algo == "Machine Learning":
            # Tentukan ukuran confusion matrix yang sama untuk semua plot
            FIGSIZE = (5, 5) 
            # === Confusion Matrix Setelah SMOTE ===
            st.subheader("Confusion Matrix Setelah SMOTE")
            # Ambil daftar model
            after_models = list(after_smt_results.keys())

            # Looping dalam bentuk pasangan (2 kolom per baris)
            for i in range(0, len(after_models), 2):
                cols = st.columns(2)  # Membuat 2 kolom

                for j in range(2):  # Looping untuk 2 kolom dalam satu baris
                    if i + j < len(after_models):  # Pastikan tidak keluar dari indeks
                        model_name = after_models[i + j]
                        result = after_smt_results[model_name]

                        with cols[j]:
                            st.write(f"### {model_name}")

                            # Membuat plot Confusion Matrix dengan ukuran tetap
                            fig, ax = plt.subplots(figsize=FIGSIZE)
                            sns.heatmap(result["confusion_matrix"], annot=True, fmt='d', cbar=False, 
                                        xticklabels=['Buzzer', 'Real User'], yticklabels=['Buzzer', 'Real User'], 
                                        ax=ax, square=True, linewidth=0.5, cmap="Oranges")

                            # Menambahkan judul dan label
                            ax.set_title(f"{model_name} Confusion Matrix", fontsize=12, fontweight='bold')
                            ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
                            ax.set_ylabel("Actual", fontsize=12, fontweight='bold')

                            # Menampilkan plot di Streamlit dalam kolom terkait
                            st.pyplot(fig)
        elif option_algo == "Deep Learning":
            # Hitung confusion matrix
            conf_matrix = confusion_matrix(y_test_encoded, y_pred_classes)

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues',
                        xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
            ax.set_title('Confusion Matrix', fontweight='bold')
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')

            # Tampilkan di Streamlit
            st.pyplot(fig)

        # CLASSIFICATION REPORT    
        st.header("Classification Report", divider='blue')
        if option_algo == "Machine Learning":
            # Ambil daftar model setelah SMOTE
            models = list(after_smt_results.keys())

            # Looping dalam bentuk pasangan (2 kolom per baris)
            for i in range(0, len(models), 2):
                cols = st.columns(2)  # Membuat 2 kolom

                for j in range(2):  # Looping untuk 2 kolom dalam satu baris
                    if i + j < len(models):  # Pastikan tidak keluar dari indeks
                        model_name = models[i + j]
                        result = after_smt_results[model_name]

                        with cols[j]:
                            st.subheader(f"📊 {model_name}")

                            # Konversi classification_report ke DataFrame
                            report_df = pd.DataFrame(result['classification_report']).transpose()

                            # Styling tabel dengan background gradient
                            styled_df = report_df.style.background_gradient(cmap="coolwarm").format(precision=3)

                            st.dataframe(styled_df)
        elif option_algo == "Deep Learning":
            # Konversi hasil prediksi ke array
            y_pred_classes = np.array(y_pred_classes)

            # Label target
            target_names = [str(label) for label in np.unique(y)]

            # Classification report sebagai dict
            cls_rpt_dict = classification_report(y_test_encoded, y_pred_classes, target_names=target_names, output_dict=True)

            # Konversi ke DataFrame
            cls_rpt_df = pd.DataFrame(cls_rpt_dict).transpose()

            # Hapus kolom support jika tidak diperlukan
            cls_rpt_df = cls_rpt_df.drop(columns=["support"], errors="ignore")

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(cls_rpt_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Classification Report", fontsize=14, fontweight='bold')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            # Tampilkan di Streamlit
            st.pyplot(fig)

        #PREDICTION
        if option_algo == "Machine Learning":
            st.header("Actual vs Prediction Comparison Random Forest and SVM Model", divider='blue')
            dfTest = pd.DataFrame({
                "Tweet": X_test,
                "Category (Data Test)": y_test,
                "RF Prediction": predictions["Random Forest"],
                "SVM Prediction": predictions["Support Vector Machine"],
            })

            st.write(dfTest.head(20))
        elif option_algo == "Deep Learning":
            st.header("Actual vs LSTM Prediction", divider='blue')
            # Convert binary labels back to original class names
            y_test_labels = [target_names[int(label)] for label in y_test_encoded]

            dfTest = pd.DataFrame({
                "Tweet": X_test,
                "Category (Data Test)": y_test,
                "LSTM Prediction": y_test_labels
            })

            st.write(dfTest.head(20))
    # ----------------------------------------------------------------------------------------------------------    
    # VADER LEXICON
    elif option == "VADER Lexicon" and run_btn:
        df = df.dropna(subset=['eng_translation'])
        x = df['eng_translation']
        y = df['Category']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        st.header("Train Test Split (80:20)", divider='blue')
        # Hitung jumlah data training dan testing
        dtrain = len(X_train)
        dtest = len(X_test)
        col1, col2 = st.columns((2))
        with col1:
            # Buat figure dan axis
            fig, ax = plt.subplots(figsize=(8, 6))

            # Buat bar chart
            bars = ax.bar(['Data Training', 'Data Testing'], [dtrain, dtest], color=['tab:blue', 'tab:orange'])

            # Tambahkan label jumlah dan persentase di atas batang
            for bar, count in zip(bars, [dtrain, dtest]):
                percentage = f"{count / (dtrain + dtest) * 100:.2f}%"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, 
                        f"{count}\n({percentage})", ha='center', va='bottom', fontsize=12)

            # Tambahkan label sumbu dan judul
            ax.set_xlabel("Kategori Data", fontsize=12, fontweight='bold')
            ax.set_ylabel("Jumlah Data", fontsize=12, fontweight='bold')
            ax.set_title("Jumlah Data Training dan Data Testing", fontsize=14, pad=15, fontweight='bold')

            st.pyplot(fig)

        st.header("Data Transformation", divider='blue')
        if option_algo == "Machine Learning":
            st.subheader("TF-IDF Vectorizing")
            tfidf = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, ngram_range=(1,1), stop_words='english')

            x_train_tfidf = tfidf.fit_transform(X_train)
            x_test_tfidf = tfidf.transform(X_test)

            tfidf_data = tfidf.fit_transform(df['eng_translation'])
            tfidf_df = pd.DataFrame(tfidf_data.toarray(), columns=tfidf.get_feature_names_out())

            top_n = 10

            top_tfidf_words = tfidf_df.apply(lambda x: x.nlargest(top_n).index.tolist(), axis=1)
            top_tfidf_values = tfidf_df.apply(lambda x: x.nlargest(top_n).values.tolist(), axis=1)

            # Buat dataframe
            top_tfidf_df = pd.DataFrame({
                "Top Words": top_tfidf_words,
                "Top Values": top_tfidf_values
            })

            st.write(top_tfidf_df.head(10))
        elif option_algo == "Deep Learning":
            st.subheader("Sequences Conversion and Padding")
            # Pisahkan majority dan minority class
            data_majority = df[df['Category'] == 'Buzzer']
            data_minority = df[df['Category'] == 'Real User']

            # Hitung rasio bias (untuk penalty tiap misklasifikasi)
            bias = data_minority.shape[0] / data_majority.shape[0]

            # Merged kembali X_train dan y_train ke dalam 1 df untuk oversampling
            train = pd.DataFrame({'eng_translation': X_train, 'Category': y_train})

            # Merged kembali X_test dan y_test ke dalam 1 df untuk pengecekan distribusi kelas
            test = pd.DataFrame({'eng_translation': X_test, 'Category': y_test})

            # ✅ Distribusi kelas sebagai tabel Streamlit
            class_distribution_df = pd.DataFrame({
                'Set': ['Training', 'Training', 'Testing', 'Testing'],
                'Category': ['Buzzer', 'Real User', 'Buzzer', 'Real User'],
                'Count': [
                    (train['Category'] == 'Buzzer').sum(),
                    (train['Category'] == 'Real User').sum(),
                    (test['Category'] == 'Buzzer').sum(),
                    (test['Category'] == 'Real User').sum()
                ]
            })
            st.markdown("**Distribusi Kelas (Data Train & Data Test):**")
            st.table(class_distribution_df)

            # Distribusi kelas sebelum oversampling
            class_distribution_before = y_train.value_counts()

            # Pisahkan kembali majority dan minority class untuk oversampling
            data_majority = train[train['Category'] == 'Buzzer']
            data_minority = train[train['Category'] == 'Real User']

            # Apply oversample untuk minority class
            data_minority_upsampled = resample(
                data_minority,
                replace=True,
                n_samples=data_majority.shape[0],
                random_state=123
            )

            # Gabungkan kembali dan acak
            data_upsampled = pd.concat([data_majority, data_minority_upsampled])
            data_upsampled = shuffle(data_upsampled, random_state=42)

            # Tokenisasi
            vocab_size = 5000
            tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
            tokenizer.fit_on_texts(df['eng_translation'].values)

            # Train set
            # Konversi teks ke dalam urutan angka (sequences)
            X_train_sequences = tokenizer.texts_to_sequences(data_upsampled['eng_translation'].values)
            max_length = max([len(seq) for seq in X_train_sequences])
            # Padding sequences agar memiliki panjang yang sama
            X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length)
            y_train_encoded = pd.get_dummies(data_upsampled['Category']).values[:, 1]

            # Test set
            # Konversi teks ke dalam urutan angka (sequences)
            X_test_sequences = tokenizer.texts_to_sequences(X_test.values)
            # Padding sequences for equalizing dimension (length)
            X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length)
            # One-hot encode labels array 1D
            y_test_encoded = pd.get_dummies(y_test).values[:, 1]

            # Menghitung distribusi kelas setelah oversampling
            class_distribution_after = data_upsampled['Category'].value_counts()

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
            st.markdown("**Dimensi Dataset Setelah Oversampling:**")
            st.table(shape_df)

        st.header("Handling Imbalance Class", divider='blue')
        if option_algo == "Machine Learning":
            st.subheader("Class Distribution Before vs After SMOTE")
            
            # Menghitung distribusi kelas sebelum menerapkan SMOTE
            class_distribution_before = y_train.value_counts()

            # Menerapkan teknik oversampling SMOTE untuk menyeimbangkan data
            smote = SMOTE(random_state=42)
            x_train_tfidf_resampled, y_train_resampled = smote.fit_resample(x_train_tfidf, y_train)

            # Menghitung distribusi kelas setelah penerapan SMOTE
            class_distribution_after = pd.Series(y_train_resampled).value_counts()

            # Membuat figure untuk visualisasi distribusi kelas
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Fungsi untuk menambahkan label angka di atas batang grafik
            def add_labels(ax, values):
                for i, v in enumerate(values):
                    ax.text(i, v + 10, str(v), ha='center', fontsize=12, color='black')

            # Grafik sebelum menerapkan SMOTE
            axes[0].bar(class_distribution_before.index.astype(str), class_distribution_before.values, color='tab:blue')
            axes[0].set_title("Distribusi Kelas Sebelum SMOTE", fontweight='bold')
            axes[0].set_xlabel("Kategori", fontweight='bold')
            axes[0].set_ylabel("Jumlah", fontweight='bold')
            add_labels(axes[0], class_distribution_before.values)

            # Grafik setelah menerapkan SMOTE
            axes[1].bar(class_distribution_after.index.astype(str), class_distribution_after.values, color='tab:orange')
            axes[1].set_title("Distribusi Kelas Setelah SMOTE", fontweight='bold')
            axes[1].set_xlabel("Kategori", fontweight='bold')
            axes[1].set_ylabel("Jumlah", fontweight='bold')
            add_labels(axes[1], class_distribution_after.values)

            plt.tight_layout()

            st.pyplot(fig)
        elif option_algo == "Deep Learning":
            st.subheader("Class Distribution Before vs After Oversampling")
            # Buat figure dan axes
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Fungsi untuk menambahkan label ke atas bar
            def add_labels(ax, values):
                for i, v in enumerate(values):
                    ax.text(i, v + 10, str(v), ha='center', fontsize=12, color='black')

            # Plot distribusi sebelum oversampling
            axes[0].bar(class_distribution_before.index.astype(str), class_distribution_before.values, color='tab:blue')
            axes[0].set_title("Class Distribution Before Oversampling", fontweight='bold')
            axes[0].set_xlabel("Kategori", fontweight='bold')
            axes[0].set_ylabel("Jumlah", fontweight='bold')
            add_labels(axes[0], class_distribution_before.values)

            # Plot distribusi setelah oversampling
            axes[1].bar(class_distribution_after.index.astype(str), class_distribution_after.values, color='tab:orange')
            axes[1].set_title("Class Distribution After Oversampling", fontweight='bold')
            axes[1].set_xlabel("Kategori", fontweight='bold')
            axes[1].set_ylabel("Jumlah", fontweight='bold')
            add_labels(axes[1], class_distribution_after.values)

            # Tata letak dan tampilkan ke Streamlit
            plt.tight_layout()
            st.pyplot(fig)    

        if option_algo == "Machine Learning":
            # Modeling before smote
            # Inisialisasi model
            models = {
                "Support Vector Machine": SVC(kernel='linear', random_state=42),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting Machine (GBM)": GradientBoostingClassifier(random_state=42),
            }

            # Dictionary untuk menyimpan hasil evaluasi model sebelum SMOTE
            before_smt_results = {}

            # Looping melalui setiap model untuk pelatihan dan evaluasi
            for model_name, model in models.items():
                # Konversi sparse matrix ke dense array jika diperlukan
                x_train_dense = x_train_tfidf.toarray() if hasattr(x_train_tfidf, "toarray") else x_train_tfidf
                x_test_dense = x_test_tfidf.toarray() if hasattr(x_test_tfidf, "toarray") else x_test_tfidf

                # Mengukur waktu pelatihan
                start_time = time.time()
                model.fit(x_train_dense, y_train)
                end_time = time.time()
                training_time = end_time - start_time
                minutes, seconds = divmod(int(training_time), 60)
                formatted_time = f"{minutes:02d} m {seconds:02d} s"

                # Melakukan prediksi
                y_pred = model.predict(x_test_dense)

                # Menyimpan hasil evaluasi
                before_smt_results[model_name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                    "training_time": formatted_time
                }

            #modeling after smote
            # Inisialisasi model
            models = {
                "Support Vector Machine": SVC(kernel='linear', random_state=42),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting Machine (GBM)": GradientBoostingClassifier(random_state=42),
            }

            # Dictionary untuk menyimpan hasil evaluasi setelah SMOTE
            after_smt_results = {}
            predictions = {}

            # Looping melalui setiap model untuk pelatihan dan evaluasi
            for model_name, model in models.items():
                # Konversi sparse matrix ke dense array setelah SMOTE jika diperlukan
                X_train_dense = x_train_tfidf_resampled.toarray() if hasattr(x_train_tfidf_resampled, "toarray") else x_train_tfidf_resampled
                X_test_dense = x_test_tfidf.toarray() if hasattr(x_test_tfidf, "toarray") else x_test_tfidf

                # Mengukur waktu pelatihan
                start_time = time.time()

                # Melatih model
                model.fit(X_train_dense, y_train_resampled)

                # Melakukan prediksi
                y_pred = model.predict(X_test_dense)

                end_time = time.time()
                training_time = end_time - start_time
                minutes, seconds = divmod(int(training_time), 60)
                formatted_time = f"{minutes:02d} m {seconds:02d} s"

                # Menyimpan hasil prediksi
                predictions[model_name] = y_pred

                # Menyimpan hasil evaluasi model
                after_smt_results[model_name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                    "training_time": formatted_time
                }
        elif option_algo == "Deep Learning":
            st.header("Visual Model Summary", divider='blue')
            # For Embedding (Hyperparameter to tune) 🔍
            input_size = np.max(X_train_padded) + 1
            # Try power of 2 (16,32,64,128,...,etc) or 50,100,200,300
            embed_dim = 64
            # Padding sequences for equalizing dimension (length)
            max_length = max([len(seq) for seq in X_train_sequences])

            # Define a model
            model = Sequential()
            # Add an embedding layer with hyperparameter based on optimal value
            model.add(Embedding(input_dim=input_size, output_dim=embed_dim, input_length=max_length))
            # Build model explicitly
            model.build(input_shape=(None, max_length))
            # Add a LSTM layer
            model.add(LSTM(64, dropout=0.2))
            # Add a dense layer with ReLU activation
            model.add(Dense(32, activation='relu'))
            # Output layer with sigmoid activation for binary crossentropy loss function
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
            col1, col2 = st.columns((2))
            # # Visualize model structure
            # plot_model(model, show_shapes=True, to_file="model_structure.png")
            # --- A. Tampilkan struktur model dalam bentuk gambar ---
            with col1:
                image = Image.open(r"C:\Users\KENT LEE\Documents\TA KENT 2025\views\model_structure.png")
                st.image(image, caption='Model Structure', use_container_width=True)

        # LSTM ONLY: EPOCH TIME SERIES    
        if option_algo == "Deep Learning":
            st.header("Epoch Time Series", divider='blue')
            # Adding weights untuk penalty dalam setiap missclassification pada sesi training
            class_weights = {0:1, 1:1.6/bias}
            # Training size (Try smaller power of 2 such as 32,64,128)
            batch_size = 32
            # Model Train
            time_series_weight = model.fit(X_train_padded, y_train_encoded, epochs=10, batch_size=batch_size, validation_data=(X_test_padded, y_test_encoded), 
                class_weight=class_weights)
            
            # Buat figure dengan dua subplot dalam satu baris
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot Accuracy
            axes[0].plot(time_series_weight.history['accuracy'], marker='o')
            axes[0].plot(time_series_weight.history['val_accuracy'], marker='s')
            axes[0].set_title('Model Accuracy', fontweight='bold')
            axes[0].set_xlabel('Epoch', fontweight='bold')
            axes[0].set_ylabel('Accuracy', fontweight='bold')
            axes[0].set_xticks(range(1, len(time_series_weight.history['accuracy']) + 1))
            # Legend di bawah plot
            axes[0].legend(['Train', 'Validation'],
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.15),
                        ncol=2)

            # Plot Loss
            axes[1].plot(time_series_weight.history['loss'], marker='o')
            axes[1].plot(time_series_weight.history['val_loss'], marker='s')
            axes[1].set_title('Model Loss', fontweight='bold')
            axes[1].set_xlabel('Epoch', fontweight='bold')
            axes[1].set_ylabel('Loss', fontweight='bold')
            axes[1].set_xticks(range(1, len(time_series_weight.history['loss']) + 1))
            # Legend di bawah plot
            axes[1].legend(['Train', 'Validation'],
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.15),
                        ncol=2)

            # Adjust layout & tampilkan di Streamlit
            plt.tight_layout(rect=[0, 0.1, 1, 1])  # Sisakan ruang di bawah untuk legends
            st.pyplot(fig)
        
        if option_algo == "Machine Learning":
            #ACCURACY X F1-SCORE COMPARATION VISUALIZATION
            st.header("Parameter Comparison for Model", divider='blue')
            col1, col2 = st.columns((2))
            with col1:
                # Ambil akurasi dari hasil sebelum dan sesudah SMOTE
                before_accuracies = {model: result['accuracy'] for model, result in before_smt_results.items()}
                after_accuracies = {model: result['accuracy'] for model, result in after_smt_results.items()}

                models = list(before_accuracies.keys())  # Pastikan urutan model sama
                before_values = [before_accuracies[model] for model in models]
                after_values = [after_accuracies[model] for model in models]

                # Menentukan posisi bar
                x = np.arange(len(models))  # Label lokasi
                width = 0.35  # Lebar bar

                # Membuat plot dengan Matplotlib
                fig, ax = plt.subplots(figsize=(8, 5))
                bars1 = ax.bar(x - width/2, before_values, width, label='Before SMOTE', color='tab:blue')
                bars2 = ax.bar(x + width/2, after_values, width, label='After SMOTE', color='tab:orange')

                # Menambahkan teks di atas setiap bar
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}',
                                ha='center', va='bottom', fontsize=10)

                # Label dan judul
                ax.set_xlabel("Model", fontweight='bold')
                ax.set_ylabel("Akurasi", fontweight='bold')
                ax.set_title("Perbandingan Akurasi Model Sebelum dan Sesudah SMOTE", fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=15)

                # Menampilkan legend di luar plot
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                plt.ylim(0, 1)  # Batas atas sumbu Y
                plt.tight_layout()

                st.subheader("Accuracy Parameter Comparison Before vs After SMOTE")
                st.pyplot(fig)

            with col2:
                # Ambil F1-Score dari hasil sebelum dan sesudah SMOTE
                before_f1_scores = {model: result['f1_score'] for model, result in before_smt_results.items()}
                after_f1_scores = {model: result['f1_score'] for model, result in after_smt_results.items()}

                models = list(before_f1_scores.keys())  # Pastikan urutan model sama
                before_values = [before_f1_scores[model] for model in models]
                after_values = [after_f1_scores[model] for model in models]

                # Menentukan posisi bar
                x = np.arange(len(models))  # Label lokasi
                width = 0.35  # Lebar bar

                # Membuat plot dengan Matplotlib
                fig, ax = plt.subplots(figsize=(8, 5))
                bars1 = ax.bar(x - width/2, before_values, width, label='Before SMOTE', color='tab:blue')
                bars2 = ax.bar(x + width/2, after_values, width, label='After SMOTE', color='tab:orange')

                # Menambahkan teks di atas setiap bar
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}',
                                ha='center', va='bottom', fontsize=10)

                # Label dan judul
                ax.set_xlabel("Model", fontweight='bold')
                ax.set_ylabel("F1-Score", fontweight='bold')
                ax.set_title("Perbandingan F1-Score Model Sebelum dan Sesudah SMOTE", fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=15)

                # Pindahkan legend ke luar chart
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                plt.ylim(0, 1)  # Batas atas sumbu Y
                plt.tight_layout()

                # **Menampilkan plot di Streamlit**
                st.subheader("F1-Score Parameter Comparison Before vs After SMOTE")
                st.pyplot(fig)
        elif option_algo == "Deep Learning":
            #ACCURACY, F1-SCORE, RECALL, PRECISION VISUALIZATION
            st.header("Parameter Comparison for LSTM", divider='blue')
            # Prediksi
            y_pred = model.predict(X_test_padded)
            y_pred_classes = (y_pred > 0.5).astype(int)

            # Hitung metrik
            accuracy = accuracy_score(y_test_encoded, y_pred_classes)
            recall = recall_score(y_test_encoded, y_pred_classes)
            precision = precision_score(y_test_encoded, y_pred_classes)
            f1 = f1_score(y_test_encoded, y_pred_classes)

            # Siapkan data untuk plot
            metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
            values = [accuracy, f1, recall, precision]

            # Buat plot bar
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(metrics, values, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])

            # Tambahkan label nilai di atas bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}',
                        ha='center', va='bottom', fontsize=12)

            # Label dan judul
            ax.set_xlabel("Metrics", fontweight='bold')
            ax.set_ylabel("Score", fontweight='bold')
            ax.set_title("Model Performance After Weighting", fontweight='bold')
            ax.set_ylim(0, 1)

            # Tampilkan di Streamlit
            st.pyplot(fig)

        #CONFUSION MATRIX
        st.header("Evaluation & Testing", divider='blue')
        if option_algo == "Machine Learning":
            # Tentukan ukuran confusion matrix yang sama untuk semua plot
            FIGSIZE = (5, 5) 
            # === Confusion Matrix Setelah SMOTE ===
            st.subheader("Confusion Matrix Setelah SMOTE")
            # Ambil daftar model
            after_models = list(after_smt_results.keys())

            # Looping dalam bentuk pasangan (2 kolom per baris)
            for i in range(0, len(after_models), 2):
                cols = st.columns(2)  # Membuat 2 kolom

                for j in range(2):  # Looping untuk 2 kolom dalam satu baris
                    if i + j < len(after_models):  # Pastikan tidak keluar dari indeks
                        model_name = after_models[i + j]
                        result = after_smt_results[model_name]

                        with cols[j]:
                            st.write(f"### {model_name}")

                            # Membuat plot Confusion Matrix dengan ukuran tetap
                            fig, ax = plt.subplots(figsize=FIGSIZE)
                            sns.heatmap(result["confusion_matrix"], annot=True, fmt='d', cbar=False, 
                                        xticklabels=['Buzzer', 'Real User'], yticklabels=['Buzzer', 'Real User'], 
                                        ax=ax, square=True, linewidth=0.5, cmap="Oranges")

                            # Menambahkan judul dan label
                            ax.set_title(f"{model_name} Confusion Matrix", fontsize=12, fontweight='bold')
                            ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
                            ax.set_ylabel("Actual", fontsize=12, fontweight='bold')

                            # Menampilkan plot di Streamlit dalam kolom terkait
                            st.pyplot(fig)
        elif option_algo == "Deep Learning":
            # Hitung confusion matrix
            conf_matrix = confusion_matrix(y_test_encoded, y_pred_classes)

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues',
                        xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
            ax.set_title('Confusion Matrix', fontweight='bold')
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')

            # Tampilkan di Streamlit
            st.pyplot(fig)

        # CLASSIFICATION REPORT    
        st.header("Classification Report", divider='blue')
        if option_algo == "Machine Learning":
            # Ambil daftar model setelah SMOTE
            models = list(after_smt_results.keys())

            # Looping dalam bentuk pasangan (2 kolom per baris)
            for i in range(0, len(models), 2):
                cols = st.columns(2)  # Membuat 2 kolom

                for j in range(2):  # Looping untuk 2 kolom dalam satu baris
                    if i + j < len(models):  # Pastikan tidak keluar dari indeks
                        model_name = models[i + j]
                        result = after_smt_results[model_name]

                        with cols[j]:
                            st.subheader(f"📊 {model_name}")

                            # Konversi classification_report ke DataFrame
                            report_df = pd.DataFrame(result['classification_report']).transpose()

                            # Styling tabel dengan background gradient
                            styled_df = report_df.style.background_gradient(cmap="coolwarm").format(precision=3)

                            st.dataframe(styled_df)
        elif option_algo == "Deep Learning":
            # Konversi hasil prediksi ke array
            y_pred_classes = np.array(y_pred_classes)

            # Label target
            target_names = [str(label) for label in np.unique(y)]

            # Classification report sebagai dict
            cls_rpt_dict = classification_report(y_test_encoded, y_pred_classes, target_names=target_names, output_dict=True)

            # Konversi ke DataFrame
            cls_rpt_df = pd.DataFrame(cls_rpt_dict).transpose()

            # Hapus kolom support jika tidak diperlukan
            cls_rpt_df = cls_rpt_df.drop(columns=["support"], errors="ignore")

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(cls_rpt_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Classification Report", fontsize=14, fontweight='bold')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            # Tampilkan di Streamlit
            st.pyplot(fig)

        #PREDICTION
        if option_algo == "Machine Learning":
            st.header("Prediction Comparison Random Forest and SVM Model", divider='blue')
            dfTest = pd.DataFrame({
                "Tweet": X_test,
                "Category (Data Test)": y_test,
                "RF Prediction": predictions["Random Forest"],
                "SVM Prediction": predictions["Support Vector Machine"],
            })
            st.write(dfTest.head(20))
        elif option_algo == "Deep Learning":
            st.header("Actual vs LSTM Prediction", divider='blue')
            # Convert binary labels back to original class names
            y_test_labels = [target_names[int(label)] for label in y_test_encoded]

            dfTest = pd.DataFrame({
                "Tweet": X_test,
                "Category (Data Test)": y_test,
                "LSTM Prediction": y_test_labels
            })

            st.write(dfTest.head(20))

