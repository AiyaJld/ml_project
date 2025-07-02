import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import ast
from collections import Counter
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, SelectFromModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
import seaborn as sns
from optimization import opt_rf, opt_tree
from sklearn.model_selection import KFold
from sklearn.base import clone
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

target = 'Rating'
# Remove rows where important columns have missing values
dataset = dataset[~dataset['méta_score'].isna()]
dataset = dataset[~dataset['production_company'].isna()]
dataset = dataset[~dataset['stars'].isna()]
dataset = dataset[~dataset['MPA'].isna()]
dataset = dataset[~dataset['Rating'].isna()]
dataset = dataset[~dataset['grossWorldWWide'].isna()]
dataset = dataset.reset_index(drop=True)

# Drop columns that are irrelevant or unnecessary for analysis
dataset = dataset.drop(columns=['gross_US_Canada', 'opening_weekend_Gross', 'budget', 'Movie Link', 'description',
                                'filming_locations', 'release_date'])

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

# Adding one-hot
dataset = pd.concat([dataset, top_writers,
                     top_director,
                     top_countries.astype(int),
                     top_productions.astype(int),
                     top_languages.astype(int)], axis=1)

# List of numerical features to scale
num_features = ['méta_score', 'Votes', 'Duration', 'Nominations',
                'Wins', 'top_actor_count']

# Split dataset into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

#took num features without the rating 
features = train_data.select_dtypes(include=[np.number]).drop(columns=[target]).columns.tolist()

X = train_data[features]
y = train_data[target]
#forward selection
def forward_selection(X, y, model, max_features=15):
    selected_features = []
    remaining_features = list(X.columns)
    best_score = np.inf
    while len(selected_features) < max_features and remaining_features:
        scores = []
        for feature in remaining_features:
            try:
                model_clone = clone(model)       #cloned clear model not to overtrain 
                model_clone.fit(X[selected_features + [feature]], y)
                preds = model_clone.predict(X[selected_features + [feature]])
                rmse = mean_squared_error(y, preds) ** 0.5
                scores.append((rmse, feature))
            except Exception as e:
                continue  

        if not scores:
            break
        scores.sort()
        #check if the score is better then prev best score
        if scores[0][0] < best_score:
            best_score, best_feature = scores[0]
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break

    return selected_features

def evaluate_model_with_cv(X, y, model, n_splits=5, max_features=15, min_votes=4):
    MODELS_WITHOUT_FS = (RandomForestRegressor, DecisionTreeRegressor)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    all_selected = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index)

        # Feature selection — that is not RandomForest или DecisionTree
        if not isinstance(model, MODELS_WITHOUT_FS):
            selected_features = forward_selection(X_train_scaled, y_train, model, max_features=max_features)
        else:
            selected_features = list(X.columns)

        all_selected.append(selected_features)

        model_clone = clone(model)
        model_clone.fit(X_train_scaled[selected_features], y_train)
        preds = model_clone.predict(X_val_scaled[selected_features])
        rmse = mean_squared_error(y_val, preds) ** 0.5
        rmse_scores.append(rmse)

        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")
        flat_list = [f for fold_feats in all_selected for f in fold_feats]

    feature_counts = Counter(flat_list)
    final_features = [feature for feature, count in feature_counts.items() if count >= min_votes]

    avg_rmse = np.mean(rmse_scores)

    return {
        "avg_rmse": avg_rmse,
        "rmse_per_fold": rmse_scores,
        "final_features": final_features
    }

x_train = (train_data.select_dtypes(include=['number', 'bool']).drop(columns=['Rating'], errors='ignore'))
y_train = train_data['Rating']
x_test = (test_data.select_dtypes(include=['number', 'bool']).drop(columns=['Rating'], errors='ignore'))
y_test = test_data['Rating']

results_lr = evaluate_model_with_cv(X, y, LinearRegression())
final_features = results_lr["final_features"]
x_train_selected = x_train[final_features]
x_test_selected = test_data[final_features]

scaler = StandardScaler()
x_train_final = scaler.fit_transform(train_data[final_features])
x_test_final = scaler.transform(test_data[final_features])

model = LinearRegression()
model.fit(x_train_final, y_train)

# Предсказываем на тесте
train_preds = model.predict(x_train_final)
test_preds = model.predict(x_test_final)

# Оцениваем качество
r2_test = r2_score(y_test, test_preds)
r2_train = r2_score(y_train, train_preds)
test_rmse = mean_squared_error(y_test, test_preds) ** 0.5
train_rmse = mean_squared_error(y_train, train_preds) ** 0.5
print(f"linear test RMSE: {test_rmse:.4f}")
print(f"linear test R2: {test_rmse:.4f}")
print(f"linear train RMSE: {train_rmse:.4f}")
print(f"linear train R2: {train_rmse:.4f}")


# DecisionTree
dt = DecisionTreeRegressor(
    max_depth=10, 
    min_samples_leaf=4, 
    random_state=42,
    min_samples_split=10)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
rmse_dt = root_mean_squared_error(y_test, y_pred)
r2_dt = r2_score(y_test, y_pred)
print(f"DT RMSE={rmse_dt:.3f},  R²={r2_dt:.3f}")

# RandomForest
rf = RandomForestRegressor(
        n_estimators=500,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
        max_depth=25,
        oob_score=True,
        bootstrap=True
    )
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
rmse_rf = root_mean_squared_error(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred)
print(f"RF   RMSE={rmse_rf:.3f},  R²={r2_rf:.3f}")





"""
train_lin[num_features] = scaler.fit_transform(train_lin[num_features])
test_lin[num_features] = scaler.transform(test_lin[num_features])

# ─────────── START FORWARD SELECTION ─────────────────────────────
# 0.  берем только числовые/булевые колонки; Rating — целевая
candidate_cols = train_lin.select_dtypes(include=['number', 'bool']).columns.drop('Rating')
y_fs = train_lin['Rating']

selected = []          # уже отобранные признаки
remaining = list(candidate_cols)
best_rmse = np.inf
min_gain = 0.001       # минимальное улучшение (силa стоп-критерия)
max_feats = 120         # верхний предел, как в примерах презентации

while remaining and len(selected) < max_feats:
    scores = []
    for col in remaining:
        feats = selected + [col]
        X = train_lin[feats]
        # 3-fold CV, метрика = RMSE (как на слайде «quality criterion»)
        rmse = -cross_val_score(LinearRegression(), X, y_fs, cv=3, scoring='neg_root_mean_squared_error').mean()
        scores.append((rmse, col))
    # выбираем фичу с наименьшим CV-RMSE
    scores.sort()
    curr_rmse, best_col = scores[0]

    # проверяем, есть ли ощутимый прирост
    if best_rmse - curr_rmse > min_gain:
        selected.append(best_col)
        remaining.remove(best_col)
        best_rmse = curr_rmse
        print(f" + {best_col:30s}  CV-RMSE → {best_rmse:.3f}")
    else:
        print("  прирост < min_gain — стоп")
        break

print(f"\nForward-selection оставил {len(selected)} признаков из {len(candidate_cols)}")
# 1.  оставляем только отобранные признаки + таргет
train_lin = train_lin[selected + ['Rating']]
test_lin = test_lin[selected + ['Rating']]
# ─────────── END FORWARD SELECTION ─────────────────────────────

num_features_fs = [c for c in train_lin.columns
                   if is_numeric_dtype(train_lin[c]) and c != 'Rating']
"""
"""
# Scaler
target_scaler = StandardScaler()
train_lin['Rating_scaled'] = target_scaler.fit_transform(train_lin['Rating'].values.reshape(-1, 1))
test_lin['Rating_scaled'] = target_scaler.transform(test_lin['Rating'].values.reshape(-1, 1))

x_train_lin = train_lin.drop(columns=['Rating', 'Rating_scaled'], errors='ignore')
y_train_lin = train_lin['Rating']
x_test_lin = test_lin.drop(columns=['Rating', 'Rating_scaled'], errors='ignore')
y_test_lin = test_lin['Rating']


print("Test RMSE:", root_mean_squared_error(y_test_lin, y_test_pred))
print("Test R2:", r2_score(y_test_lin, y_test_pred))
print("Train RMSE:", root_mean_squared_error(y_train_lin, y_train_pred))
print("Train R2:", r2_score(y_train_lin, y_train_pred))


"""'''
# Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Real values")
plt.ylabel("Predicted values")
plt.title("Real vs Predicted")
plt.show()

# Error histogram
error = y_test - y_test_pred
sns.histplot(error, bins=30, kde=True)
plt.title("Distribution of errors")
plt.xlabel("Errors")
plt.ylabel("Amount")
plt.show()

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), x_train, y_train, cv=5, scoring='neg_mean_squared_error'
)
train_rmse = (-train_scores.mean(axis=1)) ** 0.5
test_rmse = (-test_scores.mean(axis=1)) ** 0.5
plt.plot(train_sizes, train_rmse, label='Train RMSE')
plt.plot(train_sizes, test_rmse, label='Validation RMSE')
plt.xlabel("Amount of training values")
plt.ylabel("RMSE")
plt.title("Learning curve")
plt.legend()
plt.show()

# KFold
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = -scores
sns.boxplot(rmse_scores)
plt.title("RMSE Kfold")
plt.xlabel("RMSE")
plt.show()

# Ratio plot
rat_df = pd.Series(model.coef_, index=x_train.columns).sort_values()
top_rat = rat_df.reindex(rat_df.abs().sort_values(ascendingx=False).index[:20])
plt.figure(figsize=(8, 6))
top_rat.sort_values().plot(kind='barh', title='Top 10 feature of influence on the rating')
plt.xlabel('Ratio')
plt.tight_layout()
plt.show()
'''