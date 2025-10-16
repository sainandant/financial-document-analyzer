# 📄 Financial Document Analyzer

AI-powered tool for analyzing financial documents using NLP, entity extraction, sentiment analysis, and automatic summarization.

## 🌟 Features

- **Entity Recognition**: Extracts companies, people, dates, money amounts, and locations
- **Sentiment Analysis**: Analyzes document tone using BERT
- **Financial Metrics**: Automatically detects revenue, profit, growth percentages
- **Auto Summarization**: Generates concise summaries using BART
- **Word Cloud**: Visual representation of key terms
- **Interactive Visualizations**: Charts and graphs with Plotly

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** - Web interface
- **spaCy** - Named Entity Recognition
- **Transformers (Hugging Face)** - BERT (sentiment), BART (summarization)
- **PyPDF2** - PDF processing
- **Plotly** - Interactive visualizations
- **WordCloud** - Text visualization

## 📦 Installation
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install streamlit pandas numpy spacy transformers PyPDF2 plotly wordcloud matplotlib openpyxl torch

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 🚀 Usage
```bash
python -m streamlit run app.py
```

Upload PDF or TXT financial documents and get instant analysis!

## 💡 How It Works

1. **Document Upload**: Accepts PDF and TXT files
2. **Text Extraction**: Processes documents to extract raw text
3. **NLP Analysis**: 
   - spaCy extracts named entities
   - BERT analyzes sentiment
   - BART generates summaries
4. **Visualization**: Creates charts, word clouds, and metrics
5. **Insights**: Presents actionable financial intelligence

## 📊 Analysis Features

### Entity Extraction
- Organizations and companies mentioned
- Key people (executives, analysts)
- Financial amounts and figures
- Important dates and timelines
- Geographic locations

### Sentiment Analysis
- Overall document tone (positive/negative)
- Segment-by-segment breakdown
- Confidence scoring
- Visual sentiment distribution

### Financial Metrics
- Revenue figures
- Profit/loss data
- Growth percentages
- Earnings per share
- All percentage-based metrics

### Auto Summarization
- AI-generated executive summary
- Document statistics
- Key highlights extraction

## 🎯 Use Cases for Deriv

- **Earnings Report Analysis**: Quick insights from quarterly reports
- **News Sentiment**: Automated sentiment tracking for market news
- **Regulatory Documents**: Extract key information from SEC filings
- **Research Automation**: Process multiple documents efficiently
- **Client Reports**: Generate summaries for trading insights
- **Compliance**: Entity extraction for KYC/AML processes

## 🔮 Features Demonstrated

✅ **NLP Applications** - Entity extraction and document processing
✅ **Transformers** - BERT and BART models
✅ **Sentiment Analysis** - Real-time tone detection
✅ **Automation** - Batch document processing capability
✅ **Data Visualization** - Interactive charts and graphs

## 🌐 Future Enhancements

- [ ] Batch processing for multiple documents
- [ ] Comparison mode for multiple reports
- [ ] Export analysis to PDF/Excel
- [ ] Custom entity training
- [ ] Real-time news feed integration
- [ ] Multi-language support
- [ ] Advanced financial ratio calculations

## 📝 Sample Documents to Try

- Quarterly earnings reports
- Annual financial statements
- SEC 10-K/10-Q filings
- Investor presentations
- Market research reports
- News articles about companies

## 🤝 About

Built as part of AI Engineer application portfolio for Deriv, demonstrating:
- NLP expertise with spaCy and Transformers
- Production-ready document processing
- Financial technology applications
- End-to-end ML pipeline implementation

## 📄 License

MIT License - Free to use and modify

---

**Built with ❤️ using Streamlit, spaCy, BERT, and BART**