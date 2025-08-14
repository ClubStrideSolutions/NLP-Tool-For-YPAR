# 🔬 NLP Tool for YPAR (Youth Participatory Action Research)

A powerful, AI-enhanced text analysis platform designed for Youth Participatory Action Research, featuring both advanced AI capabilities and robust traditional NLP fallbacks.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

### 🤖 Dual-Mode Analysis
- **AI-Powered Mode**: Advanced analysis using OpenAI GPT models
- **Traditional NLP Mode**: Fallback using TextBlob, YAKE, LDA, and more
- Automatic mode selection based on configuration

### 📊 Comprehensive Text Analysis
- Sentiment Analysis with emotional context
- Theme Extraction and Topic Modeling
- Keyword Extraction with relevance scoring
- Named Entity Recognition
- Quote Extraction with context
- Research Insights Generation
- Interactive Q&A

### 🎨 Professional UI
- UC Berkeley Cal Colors theme (Berkeley Blue #003262, California Gold #FDB515)
- Responsive design with intuitive navigation
- Real-time analysis status indicators
- Rich visualizations and interactive charts

### 💾 Flexible Data Storage
- MongoDB integration for persistent storage
- Automatic fallback to Streamlit session state
- Local caching for improved performance
- No database required for basic operation

### 📁 File Support
- Text files (.txt, .md)
- PDF documents
- Microsoft Word (.docx)
- Multiple file batch processing
- Automatic text extraction and preprocessing
## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nlp-ypar-tool.git
cd nlp-ypar-tool
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run main_enhanced_cal.py
```

The application will open in your browser at `http://localhost:8501`

## ⚙️ Configuration

### Optional: Enable AI Features
1. Navigate to Settings page in the app
2. Enter your OpenAI API key
3. Click Save

### Optional: Enable Database
1. Navigate to Settings page
2. Enter MongoDB connection string
3. Click Save

### Environment Variables (Alternative)
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
CONNECTION_STRING=your_mongodb_connection_string_here
```

## 🎯 Usage

### 1. Upload Documents
- Click "Upload Data" in navigation
- Select one or more files
- Supported formats: TXT, PDF, DOCX, MD

### 2. Analyze Text
- Navigate to "Text Analysis"
- Select uploaded document
- Choose analysis type or run complete analysis

### 3. View Results
- Results display in organized columns
- Export options available
- Visualizations auto-generate

### 4. Advanced Features
- **RAG Analysis**: Context-aware responses with personas
- **Visualizations**: Word clouds, sentiment charts, theme networks
- **History**: Track all analyses performed

## 🔧 Fallback Systems

The application works perfectly without external dependencies:

### Without OpenAI API
- TextBlob for sentiment analysis
- YAKE for keyword extraction
- Latent Dirichlet Allocation for themes
- TF-IDF for document analysis

### Without MongoDB
- Session state storage
- In-memory caching
- Full functionality during session

## 📦 Project Structure

```
nlp-ypar-tool/
├── main_enhanced_cal.py    # Main application
├── openai_analyzer.py      # AI analysis module
├── enhanced_db_manager.py  # Database management
├── file_handlers.py        # File processing
├── ui_components.py        # UI components
├── rag_system.py          # RAG implementation
├── config.py              # Configuration management
├── requirements.txt       # Dependencies
├── assets/               # Images and resources
├── components/           # Additional components
└── utils/               # Utility functions
```

## 🧪 Testing

Run the test suite:
```bash
python test_features.py
python verify_fallbacks.py
```

## 📊 Performance

- Handles documents up to 10MB
- Batch processing for multiple files
- Cached results for repeated analyses
- Optimized for responsive UI

## 🔒 Security

- API keys stored securely
- Local data processing option
- No data transmission in fallback mode
- Session-based data isolation

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- UC Berkeley for design inspiration
- OpenAI for GPT models
- Streamlit for the framework
- Youth Participatory Action Research community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This tool is designed for research and educational purposes. Ensure compliance with your institution's data policies when processing sensitive information. 