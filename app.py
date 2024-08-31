from flask import Flask, request, jsonify
from flask import Flask, render_template, request
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from stopword import custom_stopwords
from synonym import list_synonyms
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Inisialisasi Dataset
dataset = pd.read_csv('dataset/qna_komcad.csv')
texts = dataset['Pertanyaan'].tolist()
synonym_tokens_query = []

#synonym
def get_synonyms(word, synonym_data=None):
    synonyms = set()
    if synonym_data and word in synonym_data:
        synonyms.update(synonym_data[word])
    else:
        return [word]

    return list(synonyms)

# Code Preprocessing
def text_preprocessing(text):
    if isinstance(text, float):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def text_tokenizing(text):
    tokens = word_tokenize(text)

    synonym_tokens_query.clear()
    # check synonym
    for word in tokens:
      synonyms = get_synonyms(word, synonym_data=list_synonyms)
      synonym_tokens_query.extend(synonyms)

    return tokens

def text_filtering(tokens):
    tokens = [word for word in tokens if word not in custom_stopwords]
    return tokens

def text_stemming(tokens):
    stemmer = StemmerFactory().create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens

# Text preprocessing dataset
if os.path.exists('processed_texts.pkl'):
    processed_texts = joblib.load('processed_texts.pkl')
else:
    processed_texts = []

    for text in texts:
        text = text_preprocessing(text)
        tokens = text_tokenizing(text)
        filtered_tokens = text_filtering(tokens)
        stemmed_tokens = text_stemming(filtered_tokens)
        processed_text = ' '.join(stemmed_tokens)
        processed_texts.append(processed_text)

    joblib.dump(processed_texts, 'processed_texts.pkl')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot")
def chatbot():
    query = request.args.get('msg')

    if len(query) != 0:
        # Text preprocessing query input
        processed_query = text_preprocessing(query)
        tokens_query = text_tokenizing(processed_query)
        filtered_tokens_query = text_filtering(tokens_query)
        stemmed_tokens_query = text_stemming(filtered_tokens_query)
        processed_query = ' '.join(stemmed_tokens_query)

        # tf idf vectorizer
        vectorizer = TfidfVectorizer()
        if os.path.exists('tfidf_matrix_dataset.pkl'):
            tfidf_matrix_dataset = joblib.load('tfidf_matrix_dataset.pkl')
            vectorizer.fit(processed_texts)
            tfidf_matrix_query = vectorizer.transform([processed_query])
        else:
            tfidf_matrix_dataset = vectorizer.fit_transform(processed_texts)
            tfidf_matrix_query = vectorizer.transform([processed_query])
            joblib.dump(tfidf_matrix_dataset, 'tfidf_matrix_dataset.pkl')

        # menggabungkan vektor dataset dan query
        tfidf_matrix = vstack([tfidf_matrix_dataset, tfidf_matrix_query])

        # menghitung jawaban dengan cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        # jika synonym ditemukan maka dilakukan langkah yang sama untuk query synonym
        if(synonym_tokens_query != tokens_query):
            filtered_tokens_synonym = text_filtering(synonym_tokens_query)
            stemmed_tokens_synonym = text_stemming(filtered_tokens_synonym)
            processed_synonym = ' '.join(stemmed_tokens_synonym)

            tfidf_matrix_synonym = vectorizer.transform([processed_synonym])
            tfidf_matrix_synonym = vstack([tfidf_matrix_dataset, tfidf_matrix_synonym])

            cosine_similarities_synonym = cosine_similarity(tfidf_matrix_synonym[-1], tfidf_matrix_synonym[:-1])
            
        # set nilai threshold
        similarity_threshold = 0.5

        # filter hasil berdasarkan threshold
        above_threshold_indices = np.where(cosine_similarities > similarity_threshold)[0]

        # cek apakah synonym ditemukan, supaya respon yang diberikan bisa sesuai
        if(synonym_tokens_query != tokens_query):
            if (cosine_similarities.max() > cosine_similarities_synonym.max()):
                most_similar_idx = np.argmax(cosine_similarities)
                most_similar_text = texts[most_similar_idx]
                jawaban = dataset['Jawaban'][most_similar_idx]

                response = {
                    'most_similar_question': most_similar_text,
                    'is_syonym': 'No',
                    'message': jawaban
                }

            elif cosine_similarities.max() < cosine_similarities_synonym.max():
                most_similar_idx_synonym = np.argmax(cosine_similarities_synonym)
                most_similar_text_synonym = texts[most_similar_idx_synonym]
                jawaban = dataset['Jawaban'][most_similar_idx_synonym]

                response = {
                    'most_similar_question': most_similar_text_synonym,
                    'is_syonym': 'Yes',
                    'message': jawaban
                }
        elif len(above_threshold_indices) > 0:
            most_similar_idx = np.argmax(cosine_similarities)
            most_similar_text = texts[most_similar_idx]
            jawaban = dataset['Jawaban'][most_similar_idx]

            response = {
                'most_similar_question': most_similar_text,
                'is_syonym': 'No',
                'message': jawaban
            }

        else:
            response = {
                'message': 'Maaf, saya tidak mengerti pertanyaan Anda.'
            }

        return jsonify(response)

    else:
        return jsonify({'message': 'Invalid input format. Include "question" in the request.'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)