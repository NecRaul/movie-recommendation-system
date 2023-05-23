# importing necessary libraries
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.metrics.pairwise import linear_kernel as lk

# getting current working directory
path_index = __file__.rfind('\\')
cwd = __file__[:path_index + 1]

# reading data
credits = pd.read_csv(cwd + 'tmdb_5000_credits.csv')
movies = pd.read_csv(cwd + 'tmdb_5000_movies.csv')

# printing data and its shape
# print(credits.head())
# print(credits.shape)
# print(movies.head())
# print(movies.shape)

# preparing data for union
credits_renamed = credits.rename(index=str, columns={'movie_id': 'id'})
# print(credits_renamed.head())

# union
merge = movies.merge(credits_renamed, on='id')
# print(merge.head())

# dropping unnecessary columns
cleaned = merge.drop(
    columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
# print(cleaned.head())

# TF-IDF vectorizor to remove articles from sentences
tfidf = tf(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')

# replacing NaN with a blank string
cleaned['overview'] = cleaned['overview'].fillna('')

# TF-IDF matrix construction
tfidf_matrix = tfidf.fit_transform(cleaned['overview'])

# checking for shape
# print(tfidf_matrix.shape)

# cosine similatiry matrix
cosine_sim = lk(tfidf_matrix, tfidf_matrix)

# print(cosine_sim.shape)
# print(cosine_sim[1])

# reverse map of indices and movie titles
indices = pd.Series(
    cleaned.index, index=cleaned['original_title']).drop_duplicates()


# this is where the magic happens
# you give a movie name
# we get the index of the movie from indices matrix
# we get pairwise similarity score of every movie from cosine_sim matrix
# then sort it based on the similarity score
# then pick 10 most similar ones
# get their indices
# and return their name from the cleaned list


def get_recommendations(movie_title, recommendation_count, cosine_sim=cosine_sim):
    if (movie_title in indices.index):
        idx = indices[movie_title]
    else:
        return 'Such a movie doesn\'t exist in the dataset.'
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # [1:recommendation_count + 1] is because obviously the most similar movie is the movie name we provided itself
    sim_scores = sim_scores[1:recommendation_count+1]
    movie_indices = [i[0] for i in sim_scores]
    return cleaned['original_title'].iloc[movie_indices]


while (True):
    movie_title = input('Enter movie name: ')
    recommendation_count = int(input('How many recommendations? '))
    # getting the recommendation
    print(get_recommendations(movie_title, recommendation_count))
    loopBreakInput = (input('Do you want to continue? Y/n '))
    if (loopBreakInput.lower() == 'n'):
        break