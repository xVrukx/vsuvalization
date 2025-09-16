import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:/Users/yuvraj/Projects/visualization/Netflix.csv")
df.head(3)
df.sample(10)
# Replace blank countries with the mode (most common) country
df['country'] = df['country'].fillna(df['country'].mode()[0])
df['cast'].replace(np.nan, 'No Data',inplace  = True)
df['director'].replace(np.nan, 'No Data',inplace  = True)
# Drops
df.dropna(inplace=True)
# Drop Duplicates
df.drop_duplicates(inplace= True)
df.isnull().sum()
df.info()
# Remove leading/trailing spaces
df['date_added'] = df['date_added'].str.strip()

# Convert to datetime, handling errors by coercing invalid formats to NaT
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Extract month and month name
df['month_added'] = df['date_added'].dt.month
df['month_name_added'] = df['date_added'].dt.month_name()
df['year_added'] = df['date_added'].dt.year

df.head(3)
# Helper column for various plots
df['count'] = 1

# Many productions have several countries listed - this will skew our results , we'll grab the first one mentioned

# Lets retrieve just the first country
df['first_country'] = df['country'].apply(lambda x: x.split(",")[0])
df['first_country'].head()

# Rating ages from this notebook: https://www.kaggle.com/andreshg/eda-beginner-to-expert-plotly (thank you!)

ratings_ages = {
    'TV-PG': 'Older Kids',
    'TV-MA': 'Adults',
    'TV-Y7-FV': 'Older Kids',
    'TV-Y7': 'Older Kids',
    'TV-14': 'Teens',
    'R': 'Adults',
    'TV-Y': 'Kids',
    'NR': 'Adults',
    'PG-13': 'Teens',
    'TV-G': 'Kids',
    'PG': 'Older Kids',
    'G': 'Kids',
    'UR': 'Adults',
    'NC-17': 'Adults'
}

df['target_ages'] = df['rating'].replace(ratings_ages)
df['target_ages'].unique()

# Genre

df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 

# Reducing name length

df['first_country'].replace('United States', 'USA', inplace=True)
df['first_country'].replace('United Kingdom', 'UK',inplace=True)
df['first_country'].replace('South Korea', 'S. Korea',inplace=True)
df.head()

df.to_csv(r"C:\Users\Abdo\Desktop\Cleaned_Netflix.csv", index=False)

# Palette
sns.palplot(['#221f1f', '#b20710', '#e50914','#f5f5f1'])
# Defining Netflix colors
netflix_colors = ['#221f1f', '#b20710', '#e50914', '#f5f5f1']

plt.title("Netflix brand palette ",loc='left',fontfamily='serif',fontsize=15,y=1.2)
plt.show()

x = df.groupby(['type'])['type'].count()
y = len(df)
r = ((x / y) * 100).round(2)

explode = (0.1, 0.1)  # Adjust as needed

# Creating a pie chart 
plt.figure(figsize=(10, 8))
plt.pie(
    r, 
    labels=r.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=netflix_colors[:len(r)], 
    explode=explode[:len(r)],  # Explode effect for the slices
    shadow=True,               # Adding shadow for a 3D effect
    wedgeprops=dict(edgecolor='grey')  # Adding an edge color to the wedges
)
plt.title('Ratio of Movies & TV Shows')
plt.legend(title='Type', loc='best')  # Adding a legend
plt.show()

# Converting 'duration' to numeric values (in minutes)
df['duration_numeric'] = df['duration'].str.extract('(\d+)').astype(float)

# Calculate the mean duration
mean_duration = df['duration_numeric'].mean()

# Creating a histogram of durations
plt.figure(figsize=(10, 6))
plt.hist(df['duration_numeric'].dropna(), bins=30, color='#e50914', edgecolor='black')

# Adding a vertical line for the mean duration
plt.axvline(mean_duration, color='blue', linestyle='dashed', linewidth=1, label=f'Mean Duration: {mean_duration:.2f} min')

plt.title('Distribution of Movie Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.legend()  # Show the legend with the mean duration label
plt.show()

# Creating a box plot of duration by type
plt.figure(figsize=(12, 8))
sns.boxplot(x='type', y='duration_numeric', data=df, palette=netflix_colors[:2])
plt.title('Distribution of Duration by Type')
plt.xlabel('Type')
plt.ylabel('Duration (minutes)')
plt.show()

# Exploding the genres into separate rows
genres = df['listed_in'].str.get_dummies(sep=', ')

# Summing up the number of titles per genre
genre_counts = genres.sum().sort_values(ascending=False)

# Creating a count plot for genres
plt.figure(figsize=(12, 8))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette=netflix_colors[:len(genre_counts)])
plt.xticks(rotation=90)
plt.title('Number of Titles by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Titles')
plt.show()
