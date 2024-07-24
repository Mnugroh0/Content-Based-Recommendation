# Laporan Proyek Machine Learning 
***
## Project Overview

This project focuses on analyzing and recommending restaurants using a dataset from TripAdvisor. The dataset contains restaurant data from five major states in the United States. The aim of this project is to understand the patterns and trends in restaurant reviews and, ultimately, to provide restaurant recommendations that match user preferences.

The dataset includes various attributes such as restaurant name, address, location, type of cuisine, reviews, number of reviews, comments, contact number, TripAdvisor URL, menu, and price range. By utilizing this information, we can develop a recommendation system that helps users find restaurants that best suit their preferences. The wide range of data in this dataset can help users discover more restaurants that match their tastes.

Based on this, we can create a recommendation system using content-based filtering. This method will provide users with restaurant recommendations based on their restaurant preferences.

## Business Understanding
***
### Problem Statements
Based on the project overview, the following are the issues that need to be addressed in this project:
- What kind of recommendation system would be effective, efficient, and applicable in this case?
- How can we create a restaurant recommendation system that recommends restaurants based on cuisine type, city, country, and rating?

### Goals
The objectives of this project are as follows:

- Create a restaurant recommendation system using cuisine type, city, country, and rating as features.
- Provide restaurant recommendations that users may like.

### Solution 
Solutions to achieve the project objectives include:
- Developing a recommendation model by examining the similarity of features among restaurants.
- Evaluating the recommendation results.
  
## Data Understanding
***
The dataset can be accessed at [kaggle](https://www.kaggle.com/datasets/siddharthmandgi/tripadvisor-restaurant-recommendation-data-usa/data) 

Here is the information on the dataset:
- The dataset is in CSV format.
- The dataset consists of 3062 samples with 10 features.
- All data in the dataset are objects.
- There are missing values in the dataset.

Variables in the dataset:
- Name of the Restaurant: Restaurant Name
- Street Address: Restaurant Address
- Location: Detail Location, City, Country, and Postcode
- Type of Cuisine Served: Cuisine
- Contact Number: Restaurant Contact Number
- TripAdvisor Restaurant URL: Restaurant URL
- Menu URL: Restaurant Menu URL

## Data Preparation
***
The data preparation process for this project is divided into several stages:
- Ensuring there are no duplicates and handling missing values to ensure smooth processing.
- Preprocessing data by extracting city and country from the location column, and converting rating and number of reviews to numeric columns by extracting numerical values from those columns.
- Adding features to enhance model performance, using the existing features.
- Standardizing numeric columns using MinMaxScaler.

## Modeling
***
### TF-IDF Vektorisasi
At this stage, the recommendation system will be built based on Cuisine Type, City, Country, and the Weighted Ratings of restaurants. This technique is used in recommendation systems to find the representation of important features for each restaurant.

### Cosine Similarity
In the previous stage, the correlation between restaurant names and their cuisine types was successfully identified. Now, the similarity degree between restaurant names will be calculated using the cosine similarity technique.

Cosine similarity is based on calculating the similarity between two vectors by computing the cosine of the angle between them.

$$\text{cosine similarity} = \frac{A \cdot B}{||A||_2 \times ||B||_2}$$
Where:
- $A \cdot B$ represents the dot product of vectors A and B.
- $||A||_2$ represents the Euclidean length of vector A.
- $||B||_2$ represents the Euclidean length of vector B.

## Evaluasi
***
At this stage, Precision will be used to evaluate the recommendation results. Precision can be defined as follows:

$\text{Precision} = \frac{r}{i}$

- r= total number of relevant recommendations
- i= total number of recommendations given

Using the restaurant name _Betty Lou's Seafood and Grill_ with the following table:

| Location                        | Type                                      | Reviews | No of Reviews | City           | Country | Weighted Reviews |
|---------------------------------|-------------------------------------------|---------|---------------|----------------|---------|------------------|
| San Francisco, CA 94133-3908    | Seafood, Vegetarian Friendly, Vegan Options | 4.5     | 243.0         | San Francisco  | CA      | 0.049306         |

The results obtained are as follows:
	
| Name               | Location                  | Type                                      | Reviews | No of Reviews | City           | Country | Weighted Reviews |
|--------------------|---------------------------|-------------------------------------------|---------|---------------|----------------|---------|------------------|
| Ristorante Franchino | San Francisco, CA 94133-3907 | Italian, Vegetarian Friendly, Vegan Options | 4.5     | 429.0         | San Francisco  | CA      | 0.087750         |
| Seven Hills        | San Francisco, CA 94109-3114 | Seafood, Italian, Vegetarian Friendly     | 4.5     | 923.0         | San Francisco  | CA      | 0.189854         |
| Quince             | San Francisco, CA 94133-4610 | French, Vegetarian Friendly, Vegan Options | 4.5     | 545.0         | San Francisco  | CA      | 0.111726         |
| Pacific Cafe       | San Francisco, CA 94121-1623 | American, Seafood, Gluten Free Options    | 4.5     | 241.0         | San Francisco  | CA      | 0.048893         |
| Pacific Catch      | San Francisco, CA 94123-2701 | Hawaiian, Seafood, Vegetarian Friendly    | 4.5     | 987.0         | San Francisco  | CA      | 0.203082         |


Thus, we can measure the precision. Out of the 5 recommendations given, all were relevant according to the input. Therefore, the precision obtained is 100%.
## Referensi
[cosine similarity](https://towardsdatascience.com/cosine-similarity-how-does-it-measure-the-similarity-maths-behind-and-usage-in-python-50ad30aad7db)