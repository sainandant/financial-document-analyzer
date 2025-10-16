import streamlit as st
import pandas as pd
import numpy as np
import spacy
from transformers import pipeline
import PyPDF2
import plotly.graph_objects as go
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

# Load models
@st.cache_resource
def load_nlp_models():
    download_spacy_model()
    
    # Load spaCy for entity extraction
    nlp = spacy.load("en_core_web_sm")
    
    # Load sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Load summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    return nlp, sentiment_analyzer, summarizer

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract entities using spaCy
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
    # Split text into chunks (max 512 tokens for BERT)
    max_length = 512
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i:i+max_length])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    
    sentiments = []
    for chunk in chunks[:10]:  # Limit to first 10 chunks for speed
        try:
            result = sentiment_analyzer(chunk[:512])[0]
            sentiments.append(result)
        except:
            continue
    
    return sentiments

# Extract financial metrics
def extract_financial_metrics(text):
    # Patterns for common financial metrics
    metrics = {}
    
    # Revenue patterns
    revenue_pattern = r'revenue[s]?\s*(?:of|:)?\s*\$?(\d+(?:\.\d+)?)\s*(million|billion|M|B)?'
    revenue_matches = re.findall(revenue_pattern, text, re.IGNORECASE)
    if revenue_matches:
        metrics['Revenue'] = revenue_matches
    
    # Profit patterns
    profit_pattern = r'(?:net\s+)?profit[s]?\s*(?:of|:)?\s*\$?(\d+(?:\.\d+)?)\s*(million|billion|M|B)?'
    profit_matches = re.findall(profit_pattern, text, re.IGNORECASE)
    if profit_matches:
        metrics['Profit'] = profit_matches
    
    # Growth patterns
    growth_pattern = r'growth\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*%'
    growth_matches = re.findall(growth_pattern, text, re.IGNORECASE)
    if growth_matches:
        metrics['Growth'] = [f"{g}%" for g in growth_matches]
    
    return metrics

# Generate summary
def generate_summary(text, summarizer):
    # Limit text length for summarization
    max_words = 1024
    words = text.split()[:max_words]
    text_to_summarize = " ".join(words)
    
    if len(text_to_summarize) < 100:
        return "Text too short to summarize."
    
    try:
        summary = summarizer(text_to_summarize, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

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
    - **BART** for summarization
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
    st.write("✅ Auto Summarization")
    st.write("✅ Word Cloud")

# Load models
with st.spinner("🔄 Loading AI models... (First run may take 2-3 minutes)"):
    nlp, sentiment_analyzer, summarizer = load_nlp_models()
    st.success("✅ Models loaded!")

# File upload
st.subheader("📁 Upload Document")
uploaded_file = st.file_uploader(
    "Choose a financial document (PDF or TXT)",
    type=['pdf', 'txt'],
    help="Upload earnings reports, financial statements, or any financial document"
)

if uploaded_file is not None:
    # Extract text based on file type
    with st.spinner("📖 Reading document..."):
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode('utf-8')
    
    if len(text.strip()) == 0:
        st.error("❌ Could not extract text from document. Please try another file.")
    else:
        st.success(f"✅ Extracted {len(text)} characters from document")
        
        # Show preview
        with st.expander("👁️ Preview Document Text"):
            st.text_area("First 1000 characters:", text[:1000], height=200)
        
        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏢 Entity Extraction",
            "😊 Sentiment Analysis", 
            "💰 Financial Metrics",
            "📝 Summary",
            "☁️ Word Cloud"
        ])
        
        # Tab 1: Entity Extraction
        with tab1:
            st.subheader("🏢 Named Entity Recognition")
            st.write("Identifying key entities in the document")
            
            with st.spinner("Extracting entities..."):
                entities = extract_entities(text, nlp)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Organizations
                if entities['ORG']:
                    st.write("**🏢 Organizations:**")
                    org_counts = Counter(entities['ORG'])
                    org_df = pd.DataFrame(org_counts.most_common(10), columns=['Organization', 'Mentions'])
                    st.dataframe(org_df, use_container_width=True)
                    
                    # Bar chart
                    fig_org = px.bar(org_df, x='Mentions', y='Organization', orientation='h',
                                     title='Top Organizations Mentioned')
                    st.plotly_chart(fig_org, use_container_width=True)
                else:
                    st.info("No organizations found")
                
                # People
                if entities['PERSON']:
                    st.write("**👤 People:**")
                    person_counts = Counter(entities['PERSON'])
                    person_df = pd.DataFrame(person_counts.most_common(10), columns=['Person', 'Mentions'])
                    st.dataframe(person_df, use_container_width=True)
                else:
                    st.info("No people found")
            
            with col2:
                # Money
                if entities['MONEY']:
                    st.write("**💵 Money Amounts:**")
                    money_list = list(set(entities['MONEY']))[:10]
                    st.write(", ".join(money_list))
                else:
                    st.info("No money amounts found")
                
                # Dates
                if entities['DATE']:
                    st.write("**📅 Dates:**")
                    date_counts = Counter(entities['DATE'])
                    date_df = pd.DataFrame(date_counts.most_common(10), columns=['Date', 'Mentions'])
                    st.dataframe(date_df, use_container_width=True)
                else:
                    st.info("No dates found")
                
                # Locations
                if entities['GPE']:
                    st.write("**🌍 Locations:**")
                    location_counts = Counter(entities['GPE'])
                    location_df = pd.DataFrame(location_counts.most_common(10), columns=['Location', 'Mentions'])
                    st.dataframe(location_df, use_container_width=True)
                else:
                    st.info("No locations found")
            
            # Summary metrics
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
        
        # Tab 2: Sentiment Analysis
        with tab2:
            st.subheader("😊 Sentiment Analysis")
            st.write("Analyzing the overall tone of the document")
            
            with st.spinner("Analyzing sentiment..."):
                sentiments = analyze_sentiment(text, sentiment_analyzer)
            
            if sentiments:
                # Calculate overall sentiment
                positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
                negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
                
                avg_score = np.mean([s['score'] for s in sentiments])
                
                # Display overall sentiment
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive Segments", positive_count)
                with col2:
                    st.metric("Negative Segments", negative_count)
                with col3:
                    overall = "POSITIVE 😊" if positive_count > negative_count else "NEGATIVE 😞"
                    st.metric("Overall Sentiment", overall)
                
                # Sentiment distribution
                sentiment_data = pd.DataFrame({
                    'Sentiment': ['Positive', 'Negative'],
                    'Count': [positive_count, negative_count]
                })
                
                fig_sent = px.pie(sentiment_data, values='Count', names='Sentiment',
                                  title='Sentiment Distribution',
                                  color='Sentiment',
                                  color_discrete_map={'Positive': 'green', 'Negative': 'red'})
                st.plotly_chart(fig_sent, use_container_width=True)
                
                # Confidence score
                st.write("**Analysis Confidence:**")
                st.progress(avg_score)
                st.write(f"Average confidence: {avg_score*100:.1f}%")
                
                # Detailed results
                with st.expander("📊 Detailed Segment Analysis"):
                    sent_df = pd.DataFrame(sentiments)
                    st.dataframe(sent_df, use_container_width=True)
            else:
                st.warning("Could not analyze sentiment")
        
        # Tab 3: Financial Metrics
        with tab3:
            st.subheader("💰 Financial Metrics Extraction")
            st.write("Automatically detected financial figures")
            
            metrics = extract_financial_metrics(text)
            
            if metrics:
                for metric_name, values in metrics.items():
                    st.write(f"**{metric_name}:**")
                    for item in values[:5]:
                        if isinstance(item, tuple) and len(item) >= 2:
                            value, unit = item[0], item[1] if item[1] else ''
                            st.write(f"- ${value} {unit}")
                        elif isinstance(item, tuple) and len(item) == 1:
                            st.write(f"- ${item[0]}")
                        else:
                            st.write(f"- {item}")
                    st.write("---")
            else:
                st.info("No specific financial metrics detected")
            
            # Show all percentages
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
            if percentages:
                st.write("**📊 Percentages Found:**")
                percent_counts = Counter(percentages)
                percent_df = pd.DataFrame(percent_counts.most_common(10), columns=['Percentage', 'Occurrences'])
                st.dataframe(percent_df, use_container_width=True)
        
        # Tab 4: Summary
        with tab4:
            st.subheader("📝 Document Summary")
            st.write("AI-generated summary of the document")
            
            with st.spinner("Generating summary... (this may take 30 seconds)"):
                summary = generate_summary(text, summarizer)
            
            st.success("✅ Summary Generated!")
            st.write(summary)
            
            # Key statistics
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
        
        # Tab 5: Word Cloud
        with tab5:
            st.subheader("☁️ Word Cloud")
            st.write("Visual representation of most frequent terms")
            
            with st.spinner("Generating word cloud..."):
                # Remove common words
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
                    
                    # Top words
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
    
    # Demo section
    st.subheader("🎯 What This Tool Does:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **Entity Recognition:**
        - Identifies companies, people, dates
        - Extracts money amounts and locations
        - Shows frequency of mentions
        
        **Sentiment Analysis:**
        - Analyzes document tone
        - Provides positive/negative breakdown
        - Confidence scoring
        """)
    
    with col2:
        st.write("""
        **Financial Metrics:**
        - Extracts revenue, profit figures
        - Identifies growth percentages
        - Lists all numerical data
        
        **Automation:**
        - Auto-generates summaries
        - Creates word clouds
        - Visualizes key insights
        """)

# Footer
st.write("---")
st.caption("Built with Streamlit, spaCy, BERT, and BART | Financial Document Analyzer")