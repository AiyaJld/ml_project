import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import ast
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

# Path to folder 
path = "archive-2/Data/all_years"
datasets = []

# append all CSV files from 2000 to 2025
for i in range(2000, 2026):
    file = f'{path}/merged_movies_data_{i}.csv'
    data = pd.read_csv(file)
    datasets.append(data)

# Combine all datasets into a single DataFrame
dataset = pd.concat(datasets, ignore_index=True)
dataset.to_csv('combined_dataset.csv', index=False)  
dataset = dataset.drop_duplicates()  # Remove duplicate rows

# Remove rows with missing values in important columns
dataset = dataset[~dataset['méta_score'].isna()]
dataset = dataset[~dataset['production_company'].isna()]
dataset = dataset[~dataset['stars'].isna()]
dataset = dataset[~dataset['MPA'].isna()]
dataset = dataset.reset_index(drop=True)

# Drop irrelevant or unnecessary columns
dataset = dataset.drop(columns=['gross_US_Canada','opening_weekend_Gross','budget', 'Movie Link', 'description', 'filming_locations'])

# Convert duration string into minutes
def convert_duration(duration):
    if pd.isna(duration):
        return np.nan
    hours = re.search(r'(\d+)h', duration)
    minutes = re.search(r'(\d+)m', duration)
    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    return h * 60 + m

# Convert votes with suffixes like "K" or "M" to numeric
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

# Convert gross revenue string like "$123,456,789" to integer
def convert_gross(gross):
    if pd.isna(gross):
        return np.nan
    gross = int(gross.replace('$', '').replace(',',''))
    return gross

# Replace NaN values in 'awards_content' with a default list
for i in range(len(dataset)):
    if pd.isna(dataset.at[i, 'awards_content']):
        dataset.at[i, 'awards_content'] = ['No awards']

# Apply transformations to appropriate columns
dataset['grossWorldWWide'] = dataset['grossWorldWWide'].apply(convert_gross)
dataset['Votes'] = dataset['Votes'].apply(convert_votes)
dataset['Duration'] = dataset['Duration'].apply(convert_duration)
dataset['grossWorldWWide'] = dataset['grossWorldWWide'].fillna(dataset['grossWorldWWide'].median())

# Extract award information (oscars and total wins/nominations) using regex
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

# Parse all awards information and add them as new columns
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

dataset['Oscar_nominated'] = oscar_nominated
dataset['Oscar_won'] = oscar_won
dataset['Nominations'] = nominations
dataset['Wins'] = wins

# Parse the 'stars' column from stringified lists to actual Python lists
dataset['stars'] = dataset['stars'].apply(ast.literal_eval)

# Flatten all actor names into a single list
all_actors = [actor for ls in dataset['stars'] for actor in ls]

# Count the frequency of each actor and get the top 100
actor_counts = Counter(all_actors)
top_100 = actor_counts.most_common(100)
top_actors_ls = [actor for actor, count in top_100]

# Create one-hot encoded columns for the top 100 actors
top_actors = pd.DataFrame({
    actor: dataset['stars'].map(lambda x: actor in x).astype(int)
    for actor in top_actors_ls
})

# Concatenate one-hot actor columns with the original dataset
dataset = pd.concat([dataset, top_actors], axis=1)

# Create a new feature: how many of the top 100 actors are in the movie
dataset['top_actor_count'] = dataset[top_actors_ls].sum(axis=1)

encoder = OneHotEncoder(sparse=False)
encoded_mpa = encoder.fit_transform(dataset[['MPA']])
encoded_mpa_df = pd.DataFrame(encoded_mpa, columns=encoder.get_feature_names_out(['MPA']))
dataset = dataset.drop(columns=['MPA']).reset_index(drop=True)
dataset = pd.concat([dataset, encoded_mpa_df], axis=1)

dataset['countries_origin'] = dataset['countries_origin'].apply(ast.literal_eval)
dataset['production_company'] = dataset['production_company'].apply(ast.literal_eval)
dataset['genres'] = dataset['genres'].apply(ast.literal_eval)
dataset['Languages'] = dataset['Languages'].apply(ast.literal_eval)
dataset['directors'] = dataset['directors'].apply(ast.literal_eval)
dataset['writers'] = dataset['writers'].apply(ast.literal_eval)

all_genres = set([genre for ls in dataset['genres'] for genre in ls])
genre_col = {}
for genre in all_genres:
    genre_col[f'genre_{genre}'] = dataset['genres'].apply(lambda x: int(genre in x))
genre_df = pd.DataFrame(genre_col)
dataset = pd.concat([dataset, genre_df], axis=1)

all_writers = [writer for ls in dataset['writers'] for writer in ls]
writer_counts = Counter(all_writers)
top_50 = writer_counts.most_common(50)
top_writers_ls = [writer for writer, count in top_50]
top_writers = pd.DataFrame({
    writer: dataset['writers'].map(lambda x: writer in x).astype(int)
    for writer in top_writers_ls
})

all_directors = [director for ls in dataset['directors'] for director in ls]
director_counts = Counter(all_directors)
top_40 = director_counts.most_common(40)
top_director_ls = [director for director, count in top_40]
top_director = pd.DataFrame({
    director: dataset['directors'].map(lambda x: director in x).astype(int)
    for director in top_director_ls
})

all_countries = [country for ls in dataset['countries_origin'] for country in ls]
country_counts = Counter(all_countries)
top_30 = country_counts.most_common(30)
top_countries_ls = [country for country, count in top_30]
top_countries = pd.DataFrame({
    country: dataset['countries_origin'].map(lambda x: country in x)
    for country in top_countries_ls
})

all_production = [production for ls in dataset['production_company'] for production in ls]
production_counts = Counter(all_production)
top_40_production = production_counts.most_common(40)
top_productions_ls = [production for production, count in top_40_production]
top_productions = pd.DataFrame({
    prod: dataset['production_company'].map(lambda x: prod in x)
    for prod in top_productions_ls
})

all_languages = [language for ls in dataset['Languages'] for language in ls]
languages_counts = Counter(all_languages)
top_25 = languages_counts.most_common(25)
top_languages_ls = [language for language, count in top_25]
top_languages = pd.DataFrame({
    language: dataset['Languages'].map(lambda x: language in x)
    for language in top_languages_ls
})
