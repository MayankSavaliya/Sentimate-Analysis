# Mental Health Sentiment Analysis

This project uses machine learning to analyze textual data and classify mental health status based on statements. It employs a bidirectional LSTM neural network model for sentiment analysis.

## Project Overview

The model is trained to classify statements into different mental health categories. It uses natural language processing techniques and deep learning to understand and categorize text data related to mental health.

## Features

- Text preprocessing using Tensorflow's Tokenizer
- Bidirectional LSTM neural network model
- Detailed evaluation metrics including confusion matrix and classification report
- Visualizations of training/validation accuracy and loss
- Class-wise performance analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mental-health-sentiment-analysis.git
cd mental-health-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your dataset in the `data` folder with the name "Combined Data.csv"
   - Ensure it has 'statement' and 'Mental Health Status' columns

## Usage

### Training the Model

```bash
python train_model.py
```

This will:
- Preprocess the data
- Train the model
- Generate evaluation metrics
- Save the model and visualizations

### Model Evaluation

After training, you can find:
- Confusion matrix: `models/confusion_matrix.png`
- Training history: `models/training_history.png`
- Classification report: `models/classification_report.txt`

## Project Structure

```
project/
├── data/
│   └── Combined Data.csv
├── models/
│   ├── saved_model/
│   └── (generated model files)
├── train_model.py
├── requirements.txt
├── README.md
└── LICENSE
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
