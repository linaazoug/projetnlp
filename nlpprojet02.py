import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# Fichiers CSV
train_file = "train.csv"
test_file = "test.csv"

# Charger les fichiers CSV
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Afficher les colonnes disponibles pour validation
print("Colonnes dans train.csv :", train_data.columns)
print("Colonnes dans test.csv :", test_data.columns)

# Vérifier si les colonnes spécifiques existent
required_columns = ['Class Index', 'Title', 'Description']
for column in required_columns:
    if column not in train_data.columns or column not in test_data.columns:
        print(f"La colonne '{column}' est absente du dataset. Vérifie les données.")
        exit()

# Prétraitement simple : combiner le titre et la description
def preprocess_data(df):
    df['Text'] = df['Title'] + " " + df['Description']  # Combiner 'Title' et 'Description'
    df['Text'] = df['Text'].str.replace(r'\\', '', regex=True)  # Retirer les caractères d'échappement
    df['Text'] = df['Text'].str.lower()  # Convertir en minuscules
    return df[['Class Index', 'Text']]  # Garder uniquement les colonnes nécessaires

# Appliquer le prétraitement sur train et test
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Afficher un aperçu des données prétraitées
print("Données prétraitées (train.csv) :")
print(train_data.head())
print("\nDonnées prétraitées (test.csv) :")
print(test_data.head())

# Sauvegarder les données prétraitées pour usage futur (optionnel)
train_data.to_csv("train_preprocessed.csv", index=False)
test_data.to_csv("test_preprocessed.csv", index=False)

print("\nLes fichiers prétraités ont été sauvegardés.")

# Télécharger les ressources nécessaires pour NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Initialiser le lemmatizer et la liste des stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Charger les fichiers prétraités
train_data = pd.read_csv("train_preprocessed.csv")
test_data = pd.read_csv("test_preprocessed.csv")

# Fonction pour nettoyer et prétraiter le texte
def clean_text(text):
    # Retirer les caractères spéciaux et les chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Garde uniquement les lettres et espaces
    # Convertir en minuscules
    text = text.lower()
    # Tokeniser le texte en mots
    words = text.split()
    # Supprimer les stop words et appliquer la lemmatisation
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Rejoindre les mots pour reformer le texte
    return ' '.join(words)

# Appliquer le nettoyage sur la colonne 'Text'
train_data['Cleaned_Text'] = train_data['Text'].apply(clean_text)
test_data['Cleaned_Text'] = test_data['Text'].apply(clean_text)

# Sauvegarder les fichiers nettoyés
train_data.to_csv("train_cleaned.csv", index=False)
test_data.to_csv("test_cleaned.csv", index=False)

print("Nettoyage et prétraitement terminés. Les fichiers nettoyés ont été sauvegardés.")


# Charger les fichiers nettoyés
train_data = pd.read_csv("train_cleaned.csv")
test_data = pd.read_csv("test_cleaned.csv")

# Combiner les colonnes 'Cleaned_Text' des deux fichiers pour créer un corpus global
corpus = pd.concat([train_data['Cleaned_Text'], test_data['Cleaned_Text']], axis=0)

# Tokeniser les textes (diviser chaque texte en mots)
tokenized_corpus = [text.split() for text in corpus]

# Former le modèle Word2Vec
print("Entraînement du modèle Word2Vec...")
word2vec_model = Word2Vec(
    sentences=tokenized_corpus,  # Corpus tokenisé
    vector_size=100,             # Taille des vecteurs de mots
    window=5,                    # Fenêtre de contexte
    min_count=2,                 # Ignorer les mots apparaissant moins de 2 fois
    workers=4,                   # Nombre de threads
    sg=0                         # Utilise CBOW (Continuous Bag of Words)
)

# Sauvegarder le modèle Word2Vec
word2vec_model.save("word2vec_model.model")
print("Modèle Word2Vec sauvegardé sous 'word2vec_model.model'.")

# Charger les embeddings pour vérifier les résultats
word_vectors = word2vec_model.wv


# Sauvegarder les embeddings sous forme binaire pour une utilisation rapide
word_vectors.save("word2vec_embeddings.kv")
print("Embeddings sauvegardés sous 'word2vec_embeddings.kv'.")


# Charger le modèle complet
model = Word2Vec.load("word2vec_model.model")

# Charger les embeddings
word_vectors = KeyedVectors.load("word2vec_embeddings.kv", mmap='r')  # mmap='r' optimise la mémoire

def get_word_vector(word, model, word_vectors):
    if word in model.wv:
        return model.wv[word]
    elif word in word_vectors:
        return word_vectors[word]
    else:
        return None

# Exemple : Demander à l'utilisateur de saisir un mot
word = input("Entrez un mot pour obtenir son vecteur : ")

vector = get_word_vector(word, model, word_vectors)

if vector is not None:
    print(f"Vecteur pour '{word}':\n{vector}")
else:
    print(f"Le mot '{word}' n'est pas dans le vocabulaire.")

# Exemple : Calculer la similarité entre deux mots
word1 = input("Entrez le premier mot pour calculer la similarité : ")
word2 = input("Entrez le deuxième mot pour calculer la similarité : ")

if word1 in word_vectors and word2 in word_vectors:
    similarity = word_vectors.similarity(word1, word2)
    print(f"Similarité entre '{word1}' et '{word2}' : {similarity:.2f}")
else:
    print("Un ou les deux mots ne sont pas dans le vocabulaire.")

# Exemple : Afficher des mots similaires
word = input("Entrez un mot pour voir ses mots similaires : ")

if word in word_vectors:
    similar_words = word_vectors.most_similar(word, topn=5)
    print(f"Mots similaires à '{word}' :")
    for similar_word, similarity in similar_words:
        print(f"{similar_word} (similarité : {similarity:.2f})")
else:
    print(f"Le mot '{word}' n'est pas dans le vocabulaire.")

# Charger les fichiers nettoyés
train_data = pd.read_csv("train_cleaned.csv")
test_data = pd.read_csv("test_cleaned.csv")

# Charger les embeddings pré-entraînés
word_vectors = KeyedVectors.load("word2vec_embeddings.kv", mmap='r')

# Fonction pour calculer la moyenne des embeddings des mots dans un document
def calculate_average_embedding(text, word_vectors):
    words = text.split()
    word_embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)  # Retourne un vecteur nul si aucun mot n'est présent

# Fonction pour calculer les embeddings pondérés par TF-IDF
def calculate_tfidf_weighted_embedding(text, word_vectors, tfidf_vectorizer, tfidf_matrix, doc_index):
    words = text.split()
    word_embeddings = []
    for word in words:
        if word in word_vectors and word in tfidf_vectorizer.vocabulary_:
            tfidf_weight = tfidf_matrix[doc_index, tfidf_vectorizer.vocabulary_[word]]
            word_embeddings.append(tfidf_weight * word_vectors[word])
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)

# Étape 1 : Créer une représentation TF-IDF pour le corpus global
corpus = pd.concat([train_data['Cleaned_Text'], test_data['Cleaned_Text']], axis=0)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Appliquer les techniques pour représenter les documents
train_data['Average_Embedding'] = train_data['Cleaned_Text'].apply(
    lambda text: calculate_average_embedding(text, word_vectors)
)
train_data['TFIDF_Embedding'] = [
    calculate_tfidf_weighted_embedding(text, word_vectors, tfidf_vectorizer, tfidf_matrix, idx)
    for idx, text in enumerate(train_data['Cleaned_Text'])
]

test_data['Average_Embedding'] = test_data['Cleaned_Text'].apply(
    lambda text: calculate_average_embedding(text, word_vectors)
)
test_data['TFIDF_Embedding'] = [
    calculate_tfidf_weighted_embedding(text, word_vectors, tfidf_vectorizer, tfidf_matrix, idx + len(train_data))
    for idx, text in enumerate(test_data['Cleaned_Text'])
]

# Sauvegarder les embeddings globaux
train_data.to_pickle("train_document_embeddings.pkl")
test_data.to_pickle("test_document_embeddings.pkl")

print("Les représentations des documents ont été calculées et sauvegardées.")


# Charger les embeddings globaux
train_embeddings = pd.read_pickle("train_document_embeddings.pkl")
test_embeddings = pd.read_pickle("test_document_embeddings.pkl")

# Extraire les représentations en tant que matrices numpy
train_vectors = np.stack(train_embeddings['Average_Embedding'].values)
test_vectors = np.stack(test_embeddings['Average_Embedding'].values)

# Calculer les similarités cosinus pour tout le corpus
cosine_sim_matrix = cosine_similarity(test_vectors, train_vectors)

# Exemple : Trouver les N documents les plus similaires pour chaque document de test
def find_top_n_similar_documents(similarity_matrix, n=5):
    top_n_indices = np.argsort(similarity_matrix, axis=1)[:, -n:][:, ::-1]  # Indices des N plus similaires
    return top_n_indices

# Appliquer la fonction
n = 5  # Nombre de documents à recommander
top_n_similar = find_top_n_similar_documents(cosine_sim_matrix, n=n)

# Afficher les résultats pour chaque document de test
for i, similar_indices in enumerate(top_n_similar):
    print(f"\nDocument Test {i + 1} est similaire à ces {n} documents de train :")
    for rank, index in enumerate(similar_indices):
        similarity_score = cosine_sim_matrix[i, index]
        print(f"  Rang {rank + 1}: Document Train {index + 1} (Score de similarité : {similarity_score:.4f})")




# Initialiser le modèle k-Nearest Neighbors
knn_model = NearestNeighbors(n_neighbors=n, metric='cosine')
knn_model.fit(train_vectors)

# Trouver les N documents les plus proches pour chaque document de test
distances, indices = knn_model.kneighbors(test_vectors)

# Afficher les recommandations
for i, (distance, neighbor_indices) in enumerate(zip(distances, indices)):
    print(f"\nDocument Test {i + 1} est similaire à ces {n} documents de train :")
    for rank, (dist, idx) in enumerate(zip(distance, neighbor_indices)):
        print(f"  Rang {rank + 1}: Document Train {idx + 1} (Distance cosinus : {1 - dist:.4f})")

print(train_data['Cleaned_Text'].head())
print(test_data['Cleaned_Text'].head())
