import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import ast


def convert_duration(duration):
    if pd.isna(duration):
        return np.nan
    hours = re.search(r'(\d+)h', duration)
    minutes = re.search(r'(\d+)m', duration)
    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    return h * 60 + m


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


def convert_gross(gross):
    if pd.isna(gross):
        return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(gross))
    try:
        return int(float(cleaned))
    except ValueError:
        return np.nan


df = pd.read_csv('combined_dataset.csv')

df['Duration'] = df['Duration'].apply(convert_duration)
df['Votes'] = df['Votes'].apply(convert_votes)
df['grossWorldWWide'] = df['grossWorldWWide'].apply(convert_gross)
df['genres'] = df['genres'].apply(ast.literal_eval)
numeric_cols = ['Rating', 'grossWorldWWide', 'Duration', 'Votes', 'méta_score']

# Histograms for numeric variables
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], bins=10, kde=False, color='lightBlue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Amount')
plt.tight_layout()
plt.show()

# Box plots for numeric variables
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 5, i)
    sns.boxplot(y=df[col], color='lightBlue')
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# Count plot for MPA ratings
plt.figure(figsize=(8, 5))
sns.countplot(x='MPA', data=df, color='Blue')
plt.title('Amount of Movies by MPAA Rating')
plt.xlabel('MPAA Rating')
plt.ylabel('Amount')
plt.show()

# Bar plot for genres
all_genres = [genre for genres in df['genres'] for genre in genres]
genre_counts = pd.Series(all_genres).value_counts()

total_genres = len(genre_counts)
pages = 3
genres_per_page = total_genres // pages
remainder = total_genres % pages
x_max = genre_counts.max() * 1.1
start = 0
for page in range(pages):
    end = start + genres_per_page + (1 if page < remainder else 0)
    subset = genre_counts[start:end]
    plt.figure(figsize=(10, 10))
    sns.barplot(y=subset.index, x=subset.values, color='Blue')
    plt.title(f'Genres {start + 1} to {end} of {total_genres}')
    plt.xlabel('Amount')
    plt.ylabel('Genre')
    plt.xlim(0, x_max)  # одинаковая ось X на всех страницах
    plt.tight_layout()
    plt.show()
    start = end

# Pair plot
sns.pairplot(df[numeric_cols], plot_kws={'alpha': 0.2},  height=2, aspect=1)
plt.suptitle("Pairplot on Full Dataset", y=1.02)
plt.show()
