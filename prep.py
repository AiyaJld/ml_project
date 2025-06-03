import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import ast
from collections import Counter
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Path to folder containing CSV files
path = "archive-2/Data/all_years"
datasets = []

# Loop through years 2000 to 2025 and load CSV files into a list
for i in range(2000, 2026):
    file = f'{path}/merged_movies_data_{i}.csv'
    data = pd.read_csv(file)
    datasets.append(data)

# Combine all yearly datasets into a single DataFrame
dataset = pd.concat(datasets, ignore_index=True)

# Save combined dataset to a CSV file
dataset.to_csv('combined_dataset.csv', index=False)  

# Remove duplicate rows from the combined dataset
dataset = dataset.drop_duplicates()

# Remove rows where important columns have missing values
dataset = dataset[~dataset['méta_score'].isna()]
dataset = dataset[~dataset['production_company'].isna()]
dataset = dataset[~dataset['stars'].isna()]
dataset = dataset[~dataset['MPA'].isna()]
dataset = dataset.reset_index(drop=True)

# Drop columns that are irrelevant or unnecessary for analysis
dataset = dataset.drop(columns=['gross_US_Canada', 'opening_weekend_Gross', 'budget', 'Movie Link', 'description', 'filming_locations', 'release_date'])

# Function to convert duration strings (e.g., '2h 30m') to total minutes
def convert_duration(duration):
    if pd.isna(duration):
        return np.nan
    hours = re.search(r'(\d+)h', duration)
    minutes = re.search(r'(\d+)m', duration)
    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    return h * 60 + m

# Function to convert votes with 'K' or 'M' suffixes to numeric values
def convert_votes(votes):
    if pd.isna(votes):
        return 0
    if 'K' in votes:
        num = float(re.search(r'([\d.]+)', votes).group(1)) * 1000
    elif 'M' in votes:
        num = float(re.search(r'([\d.]+)', votes).group(1)) * 1000000
    else:
        num = float(votes)
    return num

# Function to convert gross revenue strings like "$123,456,789" to integers
def convert_gross(gross):
    if pd.isna(gross):
        return np.nan
    gross = int(gross.replace('$', '').replace(',', ''))
    return gross

# Fill missing values in 'awards_content' column with a default list ['No awards']
for i in range(len(dataset)):
    if pd.isna(dataset.at[i, 'awards_content']):
        dataset.at[i, 'awards_content'] = ['No awards']

# Apply the conversion functions to appropriate columns
dataset['grossWorldWWide'] = dataset['grossWorldWWide'].apply(convert_gross)
dataset['Votes'] = dataset['Votes'].apply(convert_votes)
dataset['Duration'] = dataset['Duration'].apply(convert_duration)

# Fill missing gross revenue values with the median of the column
dataset['grossWorldWWide'] = dataset['grossWorldWWide'].fillna(dataset['grossWorldWWide'].median())

# Function to extract Oscar nominations, wins, total nominations, and total wins from awards text
def award_proc(awards):
    if not isinstance(awards, str):
        return 0, 0, 0, 0
    oscar_nominated = 0
    oscar_won = 0
    nominations = 0
    wins = 0

    find_oscar_won = re.search(r'Won (\d+) Oscars?', awards)
    if find_oscar_won:
        oscar_won = int(find_oscar_won.group(1))
    find_oscar = re.search(r'Nominated for (\d+) Oscars?', awards)
    if find_oscar:
        oscar_nominated = int(find_oscar.group(1))
    find_nominations = re.search(r'(\d+) nominations? total', awards)
    if find_nominations:
        nominations = int(find_nominations.group(1))
    find_wins = re.search(r'(\d+) wins?', awards)
    if find_wins:
        wins = int(find_wins.group(1))
    return oscar_nominated, oscar_won, nominations, wins

# Parse awards information and store in separate lists
oscar_nominated = []
oscar_won = []
nominations = []
wins = []

for text in dataset["awards_content"]:
    on, ow, n, w = award_proc(text)
    oscar_nominated.append(on)
    oscar_won.append(ow)
    nominations.append(n)
    wins.append(w)

# Add awards info as new columns to dataset
dataset['Oscar_nominated'] = oscar_nominated
dataset['Oscar_won'] = oscar_won
dataset['Nominations'] = nominations
dataset['Wins'] = wins

# Convert string representations of lists in 'stars' column to actual Python lists
dataset['stars'] = dataset['stars'].apply(ast.literal_eval)

# Flatten all actors across all movies into a single list
all_actors = [actor for ls in dataset['stars'] for actor in ls]

# Count frequency of each actor and select the top 100 most frequent actors
actor_counts = Counter(all_actors)
top_100 = actor_counts.most_common(100)
top_actors_ls = [actor for actor, count in top_100]

# Create one-hot encoded columns for presence of top 100 actors in each movie
top_actors = pd.DataFrame({
    actor: dataset['stars'].map(lambda x: actor in x).astype(int)
    for actor in top_actors_ls
})

# Add the one-hot actor columns to the original dataset
dataset = pd.concat([dataset, top_actors], axis=1)

# Create a new feature that counts how many of the top 100 actors appear in each movie
dataset['top_actor_count'] = dataset[top_actors_ls].sum(axis=1)

# One-hot encode the 'MPA' rating column using sklearn's OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_mpa = encoder.fit_transform(dataset[['MPA']])
encoded_mpa_df = pd.DataFrame(encoded_mpa, columns=encoder.get_feature_names_out(['MPA']))

# Drop original 'MPA' column and concatenate encoded columns
dataset = dataset.drop(columns=['MPA']).reset_index(drop=True)
dataset = pd.concat([dataset, encoded_mpa_df], axis=1)

# Convert stringified lists in various columns to actual lists
dataset['countries_origin'] = dataset['countries_origin'].apply(ast.literal_eval)
dataset['production_company'] = dataset['production_company'].apply(ast.literal_eval)
dataset['genres'] = dataset['genres'].apply(ast.literal_eval)
dataset['Languages'] = dataset['Languages'].apply(ast.literal_eval)
dataset['directors'] = dataset['directors'].apply(ast.literal_eval)
dataset['writers'] = dataset['writers'].apply(ast.literal_eval)

# Extract all unique genres and create one-hot encoded columns for each genre
all_genres = set([genre for ls in dataset['genres'] for genre in ls])
genre_col = {}
for genre in all_genres:
    genre_col[f'genre_{genre}'] = dataset['genres'].apply(lambda x: int(genre in x))
genre_df = pd.DataFrame(genre_col)
dataset = pd.concat([dataset, genre_df], axis=1)

# Find top 50 writers by frequency and create one-hot encoded columns for them
all_writers = [writer for ls in dataset['writers'] for writer in ls]
writer_counts = Counter(all_writers)
top_50 = writer_counts.most_common(50)
top_writers_ls = [writer for writer, count in top_50]
top_writers = pd.DataFrame({
    writer: dataset['writers'].map(lambda x: writer in x).astype(int)
    for writer in top_writers_ls
})

# Find top 40 directors by frequency and create one-hot encoded columns for them
all_directors = [director for ls in dataset['directors'] for director in ls]
director_counts = Counter(all_directors)
top_40 = director_counts.most_common(40)
top_director_ls = [director for director, count in top_40]
top_director = pd.DataFrame({
    director: dataset['directors'].map(lambda x: director in x).astype(int)
    for director in top_director_ls
})

# Find top 30 countries by frequency and create one-hot encoded columns for them
all_countries = [country for ls in dataset['countries_origin'] for country in ls]
country_counts = Counter(all_countries)
top_30 = country_counts.most_common(30)
top_countries_ls = [country for country, count in top_30]
top_countries = pd.DataFrame({
    country: dataset['countries_origin'].map(lambda x: country in x)
    for country in top_countries_ls
})

# Find top 40 production companies by frequency and create one-hot encoded columns for them
all_production = [production for ls in dataset['production_company'] for production in ls]
production_counts = Counter(all_production)
top_40_production = production_counts.most_common(40)
top_productions_ls = [production for production, count in top_40_production]
top_productions = pd.DataFrame({
    prod: dataset['production_company'].map(lambda x: prod in x)
    for prod in top_productions_ls
})

# Find top 25 languages by frequency and create one-hot encoded columns for them
all_languages = [language for ls in dataset['Languages'] for language in ls]
languages_counts = Counter(all_languages)
top_25 = languages_counts.most_common(25)
top_languages_ls = [language for language, count in top_25]
top_languages = pd.DataFrame({
    language: dataset['Languages'].map(lambda x: language in x)
    for language in top_languages_ls
})

# List of numerical features to scale
num_features = ['méta_score', 'grossWorldWWide', 'Votes', 'Duration', 'Oscar_nominated', 'Oscar_won', 'Nominations', 'Wins', 'top_actor_count', 'Year']

# Split dataset into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit scaler only on training data and transform training numerical features
train_data[num_features] = scaler.fit_transform(train_data[num_features])

# Apply the same scaler transformation to test data (no fitting)
test_data[num_features] = scaler.transform(test_data[num_features])

# Print mean and standard deviation of training data (should be ~0 and ~1)
print("Train mean:\n", train_data[num_features].mean())
print("Train std:\n", train_data[num_features].std())

# Print mean and standard deviation of test data (may differ slightly)
print("\nTest mean:\n", test_data[num_features].mean())
print("Test std:\n", test_data[num_features].std())

train_data.to_csv('processed_train_data.csv', index=False)
test_data.to_csv('processed_test_data.csv', index=False)