# 🧠 AI Mental Health Decision Support System

## Overview
A Streamlit-based application that uses a pre-trained machine learning pipeline to analyze text responses and provide insights about mental health indicators.

## Features
- **Sentence-level analysis** - Each sentence analyzed independently
- **Multi-class prediction** - Anxiety, Depression, Suicidal, Normal
- **Aggregation logic** - Combines all predictions for overall assessment
- **Risk level classification** - Low, Mild, Moderate, Moderate-High, High
- **Recommendation engine** - Tailored guidance based on results
- **Crisis resources** - Immediate support information for high-risk cases

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd mental-health-app

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Place your trained model in the root directory
# Expected file: mental_health_pipeline.pkl