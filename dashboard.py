import streamlit as st # type: ignore
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Buzzer Detection", page_icon=":robot_face:",layout="wide")


intro_page = st.Page(
    "intro.py",
    title="Introduction",
    icon=":material/account_circle:",
    default=True,
)
analysis_page = st.Page(
    "analysis.py",
    title="Data Analysis",
    icon=":material/bar_chart:",
)
detect_page = st.Page(
    "detection.py",
    title="Buzzer Detection",
    icon=":material/smart_toy:",
)
scraper_page = st.Page(
    "scraper.py",
    title="Scraper Data",
    icon=":material/library_books:",
)

# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[intro_page, scraper_page, analysis_page, detect_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [intro_page],
        "Projects": [scraper_page, analysis_page, detect_page],
    }
)


# --- SHARED ON ALL PAGES ---
st.logo("assets/buzzer.png",size="large")
st.sidebar.markdown("Made with 🐍 by [Kent](https://www.linkedin.com/in/kentleetjandra)")



# --- RUN NAVIGATION ---
pg.run()