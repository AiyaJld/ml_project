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
