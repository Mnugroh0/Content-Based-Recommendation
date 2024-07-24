# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Load Data
restaurant = pd.read_csv('TripAdvisor_RestauarantRecommendation.csv', delimiter=',')
restaurant.head(3)

restaurant['Name'].nunique()

restaurant.info()

"""
Cleaning Dataset
"""

# Drop the columns we don't need
drop_cols = ['Street Address', 'Comments', 'Contact Number', 'Trip_advisor Url', 'Menu', 'Price_Range']
restaurant = restaurant.drop(columns=drop_cols)

# Check Missing Values
restaurant.loc[restaurant.isna().any(axis=1) == True]

# Drop the columns
restaurant = restaurant.dropna()

"""
Preprocessing
"""

restaurant.loc[restaurant.duplicated()]

restaurant.loc[restaurant['Name'] == 'The Capital Grille']

restaurant = restaurant.drop_duplicates()

restaurant['City'] = restaurant['Location'].apply(lambda text: text.split(',')[0])
restaurant['Country'] = restaurant['Location'].apply(lambda text: text.split(',')[1] # Split first
                            .split(' ')[1] # Split second to get country
                            .replace(" ", "") # Remove White Space
                            )

restaurant['Reviews'] = restaurant['Reviews'].apply(lambda text: text.split(' ')[0]).astype(float)
restaurant['No of Reviews'] = restaurant['No of Reviews'].apply(lambda text: text.split(' ')[0]
                                                                .replace(',', '')
                                                                ).astype(float)

restaurant['Weighted Reviews'] = restaurant['Reviews']*restaurant['No of Reviews']

from sklearn.preprocessing import MinMaxScaler
# Standardize the Reviews and No of Reviews
mm = MinMaxScaler()

restaurant['Weighted Reviews'] = mm.fit_transform(restaurant['Weighted Reviews'].values.reshape(-1, 1))

# Set restaurant name as index
restaurant.set_index('Name', inplace=True)


"""
Modelling
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

tfidf_city = TfidfVectorizer()

stop_words = stopwords.words('english') 
tfidf_type = TfidfVectorizer(stop_words=stop_words)

tfidf_country = TfidfVectorizer()

# Fit and transform to matrix
city_matrix = tfidf_city.fit_transform(restaurant['City'])
type_matrix = tfidf_type.fit_transform(restaurant['Type'])
country_matrix = tfidf_country.fit_transform(restaurant['Country'])


# Convert the sparse matrix to a DataFrame
city_df = pd.DataFrame(city_matrix.toarray(), columns=tfidf_city.get_feature_names_out(), index=restaurant.index)
type_df = pd.DataFrame(type_matrix.toarray(), columns=tfidf_type.get_feature_names_out(), index=restaurant.index)
country_df = pd.DataFrame(country_matrix.toarray(), columns=tfidf_country.get_feature_names_out(), index=restaurant.index)

reviews_df = pd.DataFrame(index=restaurant.index, data=restaurant['Weighted Reviews'])

# Concatenate the DataFrames
main_data = pd.concat([city_df, type_df, country_df, reviews_df], axis=1)

from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(main_data)

# Convert the cosine similarity matrix to a DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=main_data.index, columns=main_data.index)


"""
Evaluation
"""

def recommend_restaurants(restaurant_name, cosine_sim_df=cosine_sim_df, restaurant_data=restaurant, top=5):
    
    index = cosine_sim_df.loc[:, restaurant_name].to_numpy().argpartition(
        range(-1, -top, -1)
    )

    closest = cosine_sim_df.columns[index[-1:-(top+2):-1]]

    closest = closest.drop(restaurant_name, errors='ignore')

    return restaurant_data.loc[closest,]


restaurant.loc["Betty Lou's Seafood and Grill",:]

# Test the recommendation system with Random Restaurant
print(recommend_restaurants("Betty Lou's Seafood and Grill"))