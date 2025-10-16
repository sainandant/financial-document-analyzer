import streamlit as st
import pandas as pd
import numpy as np
import spacy
from transformers import pipeline
import PyPDF2
import plotly.express as px
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="Financial Document Analyzer",
    page_icon="📄",
    layout="wide"
)

# Download spaCy model if not present
@st.cache_resource
def download_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        st.info("Downloading language model... This will take a minute on first run.")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Load models (lighter versions)
@st.cache_resource
def load_nlp_models():
    download_spacy_model()
    nlp = spacy.load("en_core_web_sm")
    
    # Use lighter sentiment model
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    return nlp, sentiment_analyzer

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract entities
def extract_entities(text, nlp):
    doc = nlp(text)
    entities = {
        "PERSON": [],
        "ORG": [],
        "MONEY": [],
        "DATE": [],
        "GPE": [],
        "PERCENT": []
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return entities

# Analyze sentiment
def analyze_sentiment(text, sentiment_analyzer):
    max_length = 512
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i:i+max_length])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    
    sentiments = []
    for chunk in chunks[:10]:
        try:
            result = sentiment_analyzer(chunk[:512])[0]
            sentiments.append(result)
        except:
            continue
    
    return sentiments

# Extract financial metrics
def extract_financial_metrics(text):
    metrics = {}
    
    revenue_pattern = r'revenue[s]?\s*(?:of|:)?\s*\$?(\d+(?:\.\d+)?)\s*(million|billion|M|B)?'
    revenue_matches = re.findall(revenue_pattern, text, re.IGNORECASE)
    if revenue_matches:
        metrics['Revenue'] = revenue_matches
    
    profit_pattern = r'(?:net\s+)?profit[s]?\s*(?:of|:)?\s*\$?(\d+(?:\.\d+)?)\s*(million|billion|M|B)?'
    profit_matches = re.findall(profit_pattern, text, re.IGNORECASE)
    if profit_matches:
        metrics['Profit'] = profit_matches
    
    growth_pattern = r'growth\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*%'
    growth_matches = re.findall(growth_pattern, text, re.IGNORECASE)
    if growth_matches:
        metrics['Growth'] = [f"{g}%" for g in growth_matches]
    
    return metrics

# Simple summarization (extractive)
def generate_simple_summary(text):
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if len(sentences) < 3:
        return text
    
    # Return first 3 sentences as summary
    return ". ".join(sentences[:3]) + "."

# Main App
st.title("📄 Financial Document Analyzer")
st.caption("AI-Powered Document Processing with NLP, Entity Extraction & Sentiment Analysis")

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.write("""
    This tool analyzes financial documents using:
    - **spaCy** for entity extraction
    - **BERT** for sentiment analysis
    - **Regex** for financial metrics
    """)
    
    st.write("---")
    
    st.header("💡 Supported Files")
    st.write("- PDF documents")
    st.write("- Text files (.txt)")
    
    st.write("---")
    
    st.header("🎯 Features")
    st.write("✅ Entity Recognition")
    st.write("✅ Sentiment Analysis")
    st.write("✅ Financial Metrics")
    st.write("✅ Extractive Summary")
    st.write("✅ Word Cloud")

# Load models
with st.spinner("🔄 Loading AI models..."):
    nlp, sentiment_analyzer = load_nlp_models()
    st.success("✅ Models loaded!")

# File upload
st.subheader("📁 Upload Document")
uploaded_file = st.file_uploader(
    "Choose a financial document (PDF or TXT)",
    type=['pdf', 'txt'],
    help="Upload earnings reports, financial statements, or any financial document"
)

if uploaded_file is not None:
    with st.spinner("📖 Reading document..."):
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode('utf-8')
    
    if len(text.strip()) == 0:
        st.error("❌ Could not extract text from document. Please try another file.")
    else:
        st.success(f"✅ Extracted {len(text)} characters from document")
        
        with st.expander("👁️ Preview Document Text"):
            st.text_area("First 1000 characters:", text[:1000], height=200)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏢 Entity Extraction",
            "😊 Sentiment Analysis", 
            "💰 Financial Metrics",
            "📝 Summary",
            "☁️ Word Cloud"
        ])
        
        with tab1:
            st.subheader("🏢 Named Entity Recognition")
            
            with st.spinner("Extracting entities..."):
                entities = extract_entities(text, nlp)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if entities['ORG']:
                    st.write("**🏢 Organizations:**")
                    org_counts = Counter(entities['ORG'])
                    org_df = pd.DataFrame(org_counts.most_common(10), columns=['Organization', 'Mentions'])
                    st.dataframe(org_df, use_container_width=True)
                    
                    fig_org = px.bar(org_df, x='Mentions', y='Organization', orientation='h',
                                     title='Top Organizations')
                    st.plotly_chart(fig_org, use_container_width=True)
                else:
                    st.info("No organizations found")
                
                if entities['PERSON']:
                    st.write("**👤 People:**")
                    person_counts = Counter(entities['PERSON'])
                    person_df = pd.DataFrame(person_counts.most_common(10), columns=['Person', 'Mentions'])
                    st.dataframe(person_df, use_container_width=True)
                else:
                    st.info("No people found")
            
            with col2:
                if entities['MONEY']:
                    st.write("**💵 Money Amounts:**")
                    money_list = list(set(entities['MONEY']))[:10]
                    st.write(", ".join(money_list))
                else:
                    st.info("No money amounts found")
                
                if entities['DATE']:
                    st.write("**📅 Dates:**")
                    date_counts = Counter(entities['DATE'])
                    date_df = pd.DataFrame(date_counts.most_common(10), columns=['Date', 'Mentions'])
                    st.dataframe(date_df, use_container_width=True)
                else:
                    st.info("No dates found")
                
                if entities['GPE']:
                    st.write("**🌍 Locations:**")
                    location_counts = Counter(entities['GPE'])
                    location_df = pd.DataFrame(location_counts.most_common(10), columns=['Location', 'Mentions'])
                    st.dataframe(location_df, use_container_width=True)
                else:
                    st.info("No locations found")
            
            st.write("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Organizations", len(set(entities['ORG'])))
            with col2:
                st.metric("People", len(set(entities['PERSON'])))
            with col3:
                st.metric("Money Amounts", len(set(entities['MONEY'])))
            with col4:
                st.metric("Dates", len(set(entities['DATE'])))
        
        with tab2:
            st.subheader("😊 Sentiment Analysis")
            
            with st.spinner("Analyzing sentiment..."):
                sentiments = analyze_sentiment(text, sentiment_analyzer)
            
            if sentiments:
                positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
                negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
                avg_score = np.mean([s['score'] for s in sentiments])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive Segments", positive_count)
                with col2:
                    st.metric("Negative Segments", negative_count)
                with col3:
                    overall = "POSITIVE 😊" if positive_count > negative_count else "NEGATIVE 😞"
                    st.metric("Overall Sentiment", overall)
                
                sentiment_data = pd.DataFrame({
                    'Sentiment': ['Positive', 'Negative'],
                    'Count': [positive_count, negative_count]
                })
                
                fig_sent = px.pie(sentiment_data, values='Count', names='Sentiment',
                                  title='Sentiment Distribution',
                                  color='Sentiment',
                                  color_discrete_map={'Positive': 'green', 'Negative': 'red'})
                st.plotly_chart(fig_sent, use_container_width=True)
                
                st.write("**Analysis Confidence:**")
                st.progress(avg_score)
                st.write(f"Average confidence: {avg_score*100:.1f}%")
            else:
                st.warning("Could not analyze sentiment")
        
        with tab3:
            st.subheader("💰 Financial Metrics Extraction")
            
            metrics = extract_financial_metrics(text)
            
            if metrics:
                for metric_name, values in metrics.items():
                    st.write(f"**{metric_name}:**")
                    for item in values[:5]:
                        if isinstance(item, tuple) and len(item) >= 2:
                            value, unit = item[0], item[1] if item[1] else ''
                            st.write(f"- ${value} {unit}")
                        elif isinstance(item, tuple):
                            st.write(f"- ${item[0]}")
                        else:
                            st.write(f"- {item}")
                    st.write("---")
            else:
                st.info("No specific financial metrics detected")
            
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
            if percentages:
                st.write("**📊 Percentages Found:**")
                percent_counts = Counter(percentages)
                percent_df = pd.DataFrame(percent_counts.most_common(10), columns=['Percentage', 'Occurrences'])
                st.dataframe(percent_df, use_container_width=True)
        
        with tab4:
            st.subheader("📝 Document Summary")
            
            summary = generate_simple_summary(text)
            st.success("✅ Summary Generated!")
            st.write(summary)
            
            st.write("---")
            st.write("**📊 Document Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Words", len(text.split()))
            with col2:
                st.metric("Total Characters", len(text))
            with col3:
                sentences = text.count('.') + text.count('!') + text.count('?')
                st.metric("Sentences", sentences)
        
        with tab5:
            st.subheader("☁️ Word Cloud")
            
            with st.spinner("Generating word cloud..."):
                doc = nlp(text.lower())
                words = [token.text for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 3]
                text_for_cloud = " ".join(words)
                
                if len(text_for_cloud) > 0:
                    wordcloud = WordCloud(width=800, height=400, background_color='white',
                                         colormap='viridis').generate(text_for_cloud)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    word_counts = Counter(words)
                    st.write("**🔤 Top 20 Words:**")
                    top_words_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])
                    
                    fig_words = px.bar(top_words_df, x='Frequency', y='Word', orientation='h',
                                       title='Most Frequent Words')
                    st.plotly_chart(fig_words, use_container_width=True)
                else:
                    st.warning("Not enough text to generate word cloud")

else:
    st.info("👆 Upload a document to begin analysis")
    
    st.subheader("🎯 What This Tool Does:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **Entity Recognition:**
        - Identifies companies, people, dates
        - Extracts money amounts and locations
        
        **Sentiment Analysis:**
        - Analyzes document tone
        - Positive/negative breakdown
        """)
    
    with col2:
        st.write("""
        **Financial Metrics:**
        - Extracts revenue, profit figures
        - Identifies growth percentages
        
        **Automation:**
        - Auto-generates summaries
        - Creates word clouds
        """)

st.write("---")
st.caption("Built with Streamlit, spaCy, and BERT | Financial Document Analyzer")