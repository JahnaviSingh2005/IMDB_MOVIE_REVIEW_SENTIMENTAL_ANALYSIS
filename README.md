# IMDB_MOVIE_REVIEW_SENTIMENTAL_ANALYSIS

# 🎬 IMDB Movie Review Sentiment Analysis using RNN

This project performs **sentiment analysis on IMDB movie reviews** using a **Recurrent Neural Network (RNN)** built with **TensorFlow/Keras**.
The model classifies movie reviews as **Positive** or **Negative** based on the textual content.

---

## 🚀 Project Overview

Sentiment analysis is a key Natural Language Processing (NLP) task used to understand user opinions.
In this project:

* IMDB movie reviews are preprocessed and tokenized
* Text data is converted into numerical representations using **embeddings**
* A **Simple RNN** model is trained to classify sentiment
* The trained model is saved and reused for predictions

---

## 🌟 Key Features

* Binary sentiment classification (Positive / Negative)
* Text preprocessing and tokenization
* Word embeddings for semantic understanding
* RNN-based deep learning model
* Model persistence using `.keras` format
* Prediction notebook for testing custom reviews

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **TensorFlow / Keras**
* **Natural Language Processing (NLP)**
* **Jupyter Notebook**

---

## 📂 Project Structure

```
IMDB_MOVIE_REVIEW_SENTIMENTAL_ANALYSIS/
│
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies
│
├── embedding.ipynb               # Text embedding & preprocessing
├── prediction.ipynb              # Sentiment prediction notebook
├── main.py                       # Model training script
│
└── simple_rnn_imdb.keras         # Trained RNN model
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/IMDB_MOVIE_REVIEW_SENTIMENTAL_ANALYSIS.git
```

Navigate to the project directory:

```bash
cd IMDB_MOVIE_REVIEW_SENTIMENTAL_ANALYSIS
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1️⃣ Train the Model

```bash
python main.py
```

### 2️⃣ Run Experiments & Predictions

```bash
jupyter notebook embedding.ipynb
jupyter notebook prediction.ipynb
```

---

## 🧠 Model Details

* **Model Type:** Simple Recurrent Neural Network (RNN)
* **Embedding Layer:** Converts words to dense vectors
* **Output Layer:** Sigmoid activation for binary classification
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam

---

## 🎯 Output

* **Positive Review**
* **Negative Review**

The model predicts sentiment based on the textual review provided.

---

## 🔮 Future Enhancements

* Replace RNN with **LSTM / GRU** for better performance
* Hyperparameter tuning
* Use pre-trained embeddings (GloVe, Word2Vec)
* Web app deployment (Streamlit)
* Multi-class sentiment analysis

---

## 👩‍💻 Author

**Jahnavi Singh**
B.Tech Student | NLP & Deep Learning Enthusiast

---
