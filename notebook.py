# Import necessary libraries
import numpy as np
import pandas as pd

# Display all input files in the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the data
movies = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')

# Display the first few rows of the dataframes
print(movies.head())
print(credits.head())

# Merge the datasets on 'id' column
movies = movies.merge(credits, left_on='id', right_on='movie_id', suffixes=('', '_credits'))

# Data Preprocessing
# Keep relevant columns
movies = movies[['id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew', 'release_date', 'vote_average', 'vote_count', 'popularity']]

# Define a function to extract director from the crew
def get_director(crew):
    for member in crew:
        if member['job'] == 'Director':
            return member['name']
    return np.nan

# Apply the function to the 'crew' column
import ast
movies['crew'] = movies['crew'].apply(ast.literal_eval)
movies['director'] = movies['crew'].apply(get_director)

# Extract top 3 actors from the cast
def get_top_actors(cast):
    cast = ast.literal_eval(cast)
    return [actor['name'] for actor in cast][:3]

movies['cast'] = movies['cast'].apply(get_top_actors)

# Extract genres and keywords
movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
movies['keywords'] = movies['keywords'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])

# Combine text data into a single string for vectorization
movies['combined'] = movies['overview'] + movies['keywords'].apply(lambda x: ' '.join(x)) + movies['cast'].apply(lambda x: ' '.join(x)) + movies['director'] + movies['genres'].apply(lambda x: ' '.join(x))

# Build the recommendation system using TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies.index[movies['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Test the recommendation system
print(get_recommendations('The Dark Knight'))

# Save the notebook or script as necessary
