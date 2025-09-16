# ==============================
# Netflix Dataset Analysis
# ==============================

# 1. Imports & Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Netflix color palette
NETFLIX_COLORS = ['#221f1f', '#b20710', '#e50914', '#f5f5f1']


# 2. Data Loading & Cleaning
def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Fill missing values
    df['country'] = df['country'].fillna(df['country'].mode()[0])
    df['cast'].replace(np.nan, 'No Data', inplace=True)
    df['director'].replace(np.nan, 'No Data', inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Date formatting
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['month_added'] = df['date_added'].dt.month
    df['month_name_added'] = df['date_added'].dt.month_name()
    df['year_added'] = df['date_added'].dt.year

    # Helper columns
    df['count'] = 1
    df['first_country'] = df['country'].apply(lambda x: x.split(",")[0])

    # Normalize country names
    df['first_country'].replace({
        'United States': 'USA',
        'United Kingdom': 'UK',
        'South Korea': 'S. Korea'
    }, inplace=True)

    # Map ratings to age groups
    ratings_ages = {
        'TV-PG': 'Older Kids', 'TV-MA': 'Adults', 'TV-Y7-FV': 'Older Kids',
        'TV-Y7': 'Older Kids', 'TV-14': 'Teens', 'R': 'Adults',
        'TV-Y': 'Kids', 'NR': 'Adults', 'PG-13': 'Teens',
        'TV-G': 'Kids', 'PG': 'Older Kids', 'G': 'Kids',
        'UR': 'Adults', 'NC-17': 'Adults'
    }
    df['target_ages'] = df['rating'].replace(ratings_ages)

    # Extract genres list
    df['genre'] = df['listed_in'].apply(
        lambda x: x.replace(' ,', ',').replace(', ', ',').split(',')
    )

    # Convert duration to numeric (in minutes)
    df['duration_numeric'] = df['duration'].str.extract('(\d+)').astype(float)

    return df


# 3. Visualization Functions
def plot_pie_chart(df):
    x = df.groupby(['type'])['type'].count()
    r = ((x / len(df)) * 100).round(2)
    explode = (0.1,) * len(r)

    plt.figure(figsize=(10, 8))
    plt.pie(r, labels=r.index, autopct='%1.1f%%',
            startangle=140, colors=NETFLIX_COLORS[:len(r)],
            explode=explode[:len(r)], shadow=True,
            wedgeprops=dict(edgecolor='grey'))
    plt.title('Ratio of Movies & TV Shows')
    plt.legend(title='Type', loc='best')
    plt.show()


def plot_histogram(df):
    mean_duration = df['duration_numeric'].mean()

    plt.figure(figsize=(10, 6))
    plt.hist(df['duration_numeric'].dropna(), bins=30,
             color='#e50914', edgecolor='black')
    plt.axvline(mean_duration, color='blue', linestyle='dashed',
                linewidth=1, label=f'Mean Duration: {mean_duration:.2f} min')
    plt.title('Distribution of Movie Duration')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def plot_boxplot(df):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='type', y='duration_numeric', data=df,
                palette=NETFLIX_COLORS[:2])
    plt.title('Distribution of Duration by Type')
    plt.xlabel('Type')
    plt.ylabel('Duration (minutes)')
    plt.show()


def plot_genre_count(df):
    genres = df['listed_in'].str.get_dummies(sep=', ')
    genre_counts = genres.sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=genre_counts.index, y=genre_counts.values,
                palette=NETFLIX_COLORS[:len(genre_counts)])
    plt.xticks(rotation=90)
    plt.title('Number of Titles by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Number of Titles')
    plt.show()


def plot_palette():
    sns.palplot(NETFLIX_COLORS)
    plt.title("Netflix Brand Palette", loc='left',
              fontfamily='serif', fontsize=15, y=1.2)
    plt.show()


# 4. Main Execution
if __name__ == "__main__":
    # Load data
    file_path = r"Netflix.csv"
    df = load_and_clean_data(file_path)

    # Show palette
    plot_palette()

    # Generate visualizations
    plot_pie_chart(df)
    plot_histogram(df)
    plot_boxplot(df)
    plot_genre_count(df)
