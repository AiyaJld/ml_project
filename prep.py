
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re


path = "archive-2/Data/all_years"
datasets = []

for i in range(2000, 2026):
    file = f'{path}/merged_movies_data_{i}.csv'
    data = pd.read_csv(file)
    datasets.append(data)

dataset = pd.concat(datasets, ignore_index=True)
dataset.to_csv('combined_dataset.csv', index=False)
dataset = dataset.drop_duplicates()

dataset = dataset[~dataset['méta_score'].isna()].reset_index(drop=True)
print(len(dataset))
print(dataset['méta_score'].dtype)


def convert_duration(duration):
    if pd.isna(duration):
        return np.nan
    hours = re.search(r'(\d+)h', duration)
    minutes = re.search(r'(\d+)m', duration)
    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    return h * 60 + m

def convert_votes(votes):
    if 'K' in votes:
        num = float(re.search(r'([\d.]+)', votes).group(1)) * 1000
    elif 'M' in votes:
        num = float(re.search(r'([\d.]+)', votes).group(1)) * 1000000
    else:
        num = float(votes)
    return num

def convert_gross(gross):
    if pd.isna(gross):
        return np.nan
    gross = int(gross.replace('$', '').replace(',',''))
    return gross

dataset['grossWorldWWide'] = dataset['grossWorldWWide'].apply(convert_gross)
dataset['Votes'] = dataset['Votes'].apply(convert_votes)
dataset['Duration'] = dataset['Duration'].apply(convert_duration)

dataset['Duration'] = dataset['Duration'].fillna(dataset['Duration'].median())
dataset['MPA'] = dataset['MPA'].fillna('Not Rated')
dataset['stars'] = dataset['stars'].fillna('Unknown')
dataset['description'] = dataset['description'].fillna('Not Given')
dataset[['countries_origin','production_company','filming_locations']] = dataset[['countries_origin','production_company','filming_locations']].fillna('Unknown')
dataset['awards_content'] = dataset['awards_content'].fillna('No Awards')
dataset['grossWorldWWide'] = dataset['grossWorldWWide'].fillna(dataset['grossWorldWWide'].median())

