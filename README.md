# Sentiment Analysis with Neural Networks

## Overview
This project implements **Feedforward Neural Network (FFNN)** and **Recurrent Neural Network (RNN)** models to perform **5-class sentiment analysis** on Yelp reviews. The goal is to predict the star rating (1 to 5) based on the review text.

## Repository Structure
```
📂 sentiment-analysis-project
│── ffnn.py               # Implementation of Feedforward Neural Network
│── rnn.py                # Implementation of Recurrent Neural Network
│── word_embedding.pkl    # Pretrained word embeddings for RNN
│── requirements.txt      # List of required dependencies
│── train.json            # Training dataset
│── val.json              # Validation dataset
│── test.json             # Test dataset (if applicable)
│── report.pdf            # Project report
│── README.md             # This file
```

## Requirements
- Python 3.8
- PyTorch 1.10.1
- NumPy
- tqdm
- argparse
- pickle

To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
### Train FFNN Model
```bash
python ffnn.py --hidden_dim 100 --epochs 10 --train_data train.json --val_data val.json
```

### Train RNN Model
```bash
python rnn.py --hidden_dim 100 --epochs 10 --train_data train.json --val_data val.json
```

## Implementation Details
- **FFNN:** Uses bag-of-words representation.
- **RNN:** Uses pretrained word embeddings.
- **Loss Function:** Negative Log Likelihood Loss (NLLLoss).
- **Optimizer:** SGD for FFNN, Adam for RNN.
- **Training Strategy:** Mini-batch gradient descent with batch size = 16.

## Results & Performance
| Model  | Training Accuracy | Validation Accuracy |
|--------|------------------|--------------------|
| FFNN   | XX%             | XX%                |
| RNN    | XX%             | XX%                |

## Contributions
- Sai Sathwik Annabathula - Implementation & Report

## License
This project is for educational purposes.
