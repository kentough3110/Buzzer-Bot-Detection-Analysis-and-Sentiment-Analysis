# Import Library
import streamlit as st # type: ignore
import plotly.express as px
import pandas as pd
import os
import subprocess
import io
import time
import warnings
warnings.filterwarnings('ignore')

# Reset session state when the page is refreshed
if "last_page" not in st.session_state:
    st.session_state.last_page = st.query_params

if st.query_params != st.session_state.last_page:
    st.session_state.clear()
    st.session_state.last_page = st.query_params

st.title(" 🌐 HTML Data Scraper")
st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>',unsafe_allow_html=True)

# Step 1: Enter Authentication Token
st.markdown('### 1️⃣ **Enter an authentication token before scraping**')
twitter_auth_token = st.text_input("Enter your Twitter authentication token:", type="password")
st.info("ℹ️ - **Example:** 45aeu36842a70aoeis08fb12220060")

if st.button("Run Authentication"):
    if twitter_auth_token:
        with st.spinner("Validating token..."):
            time.sleep(2)  # Simulating validation
        st.session_state.step1_done = True
        st.success("🎉 - Authentication token entered successfully!")
    else:
        st.error("⚠️ - Please enter a valid authentication token.")

# Step 2: Enter Keyword for Scraping (Only appears after Step 1 is completed)
if st.session_state.get("step1_done", False):
    st.markdown('### 2️⃣ **Enter a keyword to scrape**')
    # Input keyword untuk scrap
    search_keyword = st.text_input("Enter search keyword:")
    st.info("ℹ️ - **Example:** doktif since:2024-01-19 until:2025-02-17 lang:id")

    # Pilih limit data yang diinginkan
    limit = st.slider("Select the limit for tweets:", min_value=10, max_value=1000, value=100, step=10)

    # Input nama file
    filename = st.text_input("Enter filename:")
    st.info("ℹ️ - **Example:** tweets.csv (with extension)")

    if st.button("Run Scraping"):
        if search_keyword:
            with st.spinner("Scraping tweets..."):
                command = f"npx -y tweet-harvest@2.6.1 -o \"{filename}\" -s \"{search_keyword}\" --tab \"LATEST\" -l {limit} --token {twitter_auth_token}"
                os.system(command)
                time.sleep(5)
                
                # Simulasi hasil scraping sebagai DataFrame
                data = pd.read_csv(rf"C:\Users\KENT LEE\Documents\TA KENT 2025\tweets-data\{filename}")
                df = pd.DataFrame(data)
                st.session_state.scraped_data = df
                
            st.session_state.step2_done = True
            st.success("🎉 - Scraping completed!")
        else:
            st.error("⚠️ - Please enter a valid search keyword.")

# Step 3: Show Result Scraping Data (Only appears after Step 2 is completed)
if st.session_state.get("step2_done", False) and st.session_state.scraped_data is not None:
    st.markdown('### 3️⃣ **Result scraping data**')
    st.dataframe(st.session_state.scraped_data)
    