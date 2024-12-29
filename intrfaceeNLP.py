import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Charger les données nettoyées et le modèle Word2Vec
@st.cache_resource
def load_data_and_model():
    # Charger les fichiers nettoyés
    train_data = pd.read_csv("train_cleaned.csv")  # Contient une colonne 'Cleaned_Text'
    test_data = pd.read_csv("test_cleaned.csv")    # Contient une colonne 'Cleaned_Text'

    # Charger le modèle Word2Vec pré-entraîné
    word2vec_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    return train_data, test_data, word2vec_model

# Calcul de l'embedding moyen pour un texte donné
def get_average_word2vec(text, model, vector_size=300):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Visualisation des similarités sous forme de diagramme en barres
def plot_similarity(similarity_scores, top_n_indices, train_data):
    top_scores = similarity_scores[top_n_indices]
    top_texts = train_data['Cleaned_Text'].iloc[top_n_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_scores, y=top_texts, palette="viridis")
    plt.xlabel("Scores de Similarité")
    plt.ylabel("Documents")
    plt.title("Top 5 Documents les Plus Similaires")
    st.pyplot(plt)

# Visualisation des distributions des mots dans le corpus
def plot_word_distribution(train_data):
    all_words = " ".join(train_data['Cleaned_Text'])
    word_counts = Counter(all_words.split())
    common_words = word_counts.most_common(20)
    words, counts = zip(*common_words)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), palette="coolwarm")
    plt.xlabel("Fréquence")
    plt.ylabel("Mots")
    plt.title("20 Mots les Plus Fréquents dans le Corpus")
    st.pyplot(plt)

# Nuage de mots pour le corpus
def plot_wordcloud(train_data):
    all_words = " ".join(train_data['Cleaned_Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Nuage de Mots du Corpus")
    st.pyplot(plt)

# Charger les données et modèles
train_data, test_data, w2v_model = load_data_and_model()

# Calculer les embeddings pour le dataset de train
train_data["Embeddings"] = train_data["Cleaned_Text"].apply(
    lambda x: get_average_word2vec(x, w2v_model) if isinstance(x, str) else np.zeros(300)
)
train_vectors = np.vstack(train_data["Embeddings"].values)

# Interface utilisateur Streamlit
st.title("Système de Recommandation - Word2Vec")
st.markdown("Recherchez des documents similaires en utilisant le modèle Word2Vec.")

# Choix de méthode
option = st.radio("Choisissez une méthode :", ("Saisir un texte", "Utiliser un fichier existant"))

if option == "Saisir un texte":
    user_text = st.text_area("Entrez votre texte :", height=200)
    if st.button("Lancer la recherche") and user_text.strip():
        user_embedding = get_average_word2vec(user_text.strip(), w2v_model)
        similarity_scores = cosine_similarity([user_embedding], train_vectors).flatten()
        top_n_indices = similarity_scores.argsort()[-5:][::-1]
        
        st.subheader("Recommandations :")
        for rank, idx in enumerate(top_n_indices):
            st.write(f"**Rang {rank + 1}** : {train_data['Cleaned_Text'].iloc[idx]} (Score : {similarity_scores[idx]:.4f})")
        
        # Ajouter la visualisation des similarités
        st.subheader("Visualisation des Similarités")
        plot_similarity(similarity_scores, top_n_indices, train_data)

elif option == "Utiliser un fichier existant":
    document_idx = st.number_input("Index du document (1-10) :", min_value=1, max_value=10, step=1)
    if st.button("Lancer la recherche"):
        selected_text = test_data["Cleaned_Text"].iloc[document_idx - 1]
        user_embedding = get_average_word2vec(selected_text, w2v_model)
        similarity_scores = cosine_similarity([user_embedding], train_vectors).flatten()
        top_n_indices = similarity_scores.argsort()[-5:][::-1]
        
        st.subheader("Recommandations :")
        for rank, idx in enumerate(top_n_indices):
            st.write(f"**Rang {rank + 1}** : {train_data['Cleaned_Text'].iloc[idx]} (Score : {similarity_scores[idx]:.4f})")
        
        # Ajouter la visualisation des similarités
        st.subheader("Visualisation des Similarités")
        plot_similarity(similarity_scores, top_n_indices, train_data)

# Section pour visualiser les données du corpus
st.sidebar.title("Analyse du Corpus")
if st.sidebar.button("Afficher la Distribution des Mots"):
    st.subheader("Distribution des Mots")
    plot_word_distribution(train_data)

if st.sidebar.button("Afficher le Nuage de Mots"):
    st.subheader("Nuage de Mots")
    plot_wordcloud(train_data)
