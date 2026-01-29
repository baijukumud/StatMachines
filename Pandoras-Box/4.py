import kagglehub

path = kagglehub.dataset_download("luisreimberg/ratingscsv")
print("Path to dataset files:", path)

import pandas as pd
df = pd.read_csv(f'{path}/ratings.csv')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

interaction_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

scaler = StandardScaler(with_mean=False)
interaction_matrix_scaled = scaler.fit_transform(interaction_matrix)

user_similarity = cosine_similarity(interaction_matrix_scaled)
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index,
columns=interaction_matrix.index)

def recommend(user_id, k=5):
  similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:k+1]
  similar_users_ratings = interaction_matrix.loc[similar_users.index]
  weighted_ratings = similar_users_ratings.T.dot(similar_users)
  user_rated = interaction_matrix.loc[user_id]
  recommendations = weighted_ratings[user_rated == 0].sort_values(ascending=False).head(k)
  return recommendations.index.tolist()

user_id = int(input("Enter your input: "))
recommendations = recommend(user_id)
print(f"Recommendations for User {user_id}: {recommendations}")
