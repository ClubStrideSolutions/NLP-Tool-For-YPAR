# NLP Tool for YPAR

An interactive Natural Language Processing tool designed for Youth Participatory Action Research (YPAR), built with Streamlit and OpenAI's GPT-4.

## Features

- Text Analysis and Processing
- Topic Modeling
- Quote Extraction
- Theme Network Visualization
- Named Entity Recognition
- Ethics & Bias Analysis
- Analysis History Tracking

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd nlp-tool-for-ypar
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_openai_api_key
CONNECTION_STRING=your_mongodb_connection_string
```

4. Run the application:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Add the following secrets in Streamlit Cloud:
   - OPENAI_API_KEY
   - CONNECTION_STRING

## Dependencies

- Python 3.8+
- Streamlit
- OpenAI API
- MongoDB
- Other dependencies listed in requirements.txt

## Usage

1. Upload text files (.txt), Word documents (.docx), or Excel spreadsheets (.xlsx)
2. Select analysis features from the sidebar
3. View results and visualizations
4. Track analysis history

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 