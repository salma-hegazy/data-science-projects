import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Move import to the top
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('IMDb Movies.csv', encoding='latin1')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


df = df.drop('Duration', axis=1)

print(df.isnull().sum())
print(df.dtypes)

df['Actor 1'] = df['Actor 1'].fillna('Unknown')
df['Actor 2'] = df['Actor 2'].fillna('Unknown')
df['Actor 3'] = df['Actor 3'].fillna('Unknown')
df['Genre'] = df['Genre'].fillna('Unknown')

df['Rating'] = df['Rating'].fillna(df['Rating'].median())

df['Year'] = df['Year'].fillna(df['Year'].median())

df['Votes'] = df['Votes'].fillna(df['Votes'].mode().dropna().values[0])

df = df.dropna()
print(df.isnull().sum())
print(df.info())


movies_per_Directors = df['Director'].value_counts().reset_index()
movies_per_Directors.columns = ['Director', 'movies_count']
top_10_Directors = movies_per_Directors.head(10)

fig = px.bar(
    top_10_Directors,
    x='movies_count',
    y='Director',
    orientation='h',
    title='Top 10 Directors by Number of Movies',
    labels={'movies_count': 'Number of Movies', 'Director': 'Director'},
    color='movies_count',
    color_continuous_scale='Blues'
)
fig.update_layout(
    xaxis_title='Number of Movies',
    yaxis_title='Director',
    yaxis=dict(autorange="reversed"),
    template='plotly_white'
)
fig.show()

if 'Year' in df.columns and 'Genre' in df.columns and 'Votes' in df.columns:
    genre_popularity = df.groupby(["Year", "Genre"])["Votes"].sum().reset_index()
    fig = px.bar(
        genre_popularity,
        x="Year",
        y="Votes",
        color="Genre",
        title="Most Popular Genre Over the Years",
        labels={"Votes": "Total Votes", "Year": "Year"},
        barmode="stack"
    )
    fig.show()


if 'Rating' in df.columns:
    movies_per_Rating = df['Rating'].value_counts().reset_index()
    movies_per_Rating.columns = ['Rating', 'movies_count']
    top_movies = movies_per_Rating.head(10)

    fig = px.bar(
        top_movies,
        x='movies_count',
        y='Rating',
        orientation='h',
        title='Top Movies by Rating',
        labels={'movies_count': 'Number of Movies', 'Rating': 'Rating'},
        color='movies_count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_title='Number of Movies',
        yaxis_title='Rating',
        yaxis=dict(autorange="reversed"),
        template='plotly_white'
    )
    fig.show()