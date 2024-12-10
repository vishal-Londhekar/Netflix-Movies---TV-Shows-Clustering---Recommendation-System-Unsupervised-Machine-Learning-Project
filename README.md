# **Netflix Movies and TV Shows Clustering Machine Learning Project**
![images (2)](https://github.com/user-attachments/assets/338d1b55-1880-473d-90fa-ea4af4fa79c7)

The primary goal of this project was to analyze a dataset of TV shows and movies available on Netflix as of 2019. Using natural language processing (NLP) techniques, the objective was to group content into meaningful clusters and develop a recommendation system to improve user experience and reduce subscriber churn. With Netflix serving over 200 million subscribers worldwide, enhancing its offerings is critical to maintaining its leadership in the streaming industry.

The project began by addressing data quality issues, such as handling missing values and processing nested columns (e.g., director, cast, listed_in, and country). These steps ensured a cleaner and more structured dataset for analysis.

To categorize content effectively, the rating attribute was binned into groups, such as adult, children's, family-friendly, and not rated. This classification supports the delivery of recommendations tailored to viewer preferences and age appropriateness.

Exploratory data analysis (EDA) was conducted to uncover patterns, distributions, and relationships within the dataset. This analysis provided insights into the diversity and characteristics of Netflix's content.

NLP techniques were employed to tokenize, preprocess, and vectorize textual attributes, including director, cast, country, genre, rating, and description. By leveraging the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer, the project quantified textual data to identify similarities among movies and TV shows.

Dimensionality reduction was performed using Principal Component Analysis (PCA) to optimize computational efficiency while retaining meaningful data representation.

For clustering, two algorithms were implemented: K-Means and Agglomerative Hierarchical Clustering. The optimal number of clusters was determined using evaluation methods like the Elbow Method, Silhouette Score, and dendrogram analysis, ensuring well-defined groupings.

A content-based recommender system was then developed using a cosine similarity matrix. This system analyzed user viewing histories to deliver personalized content suggestions based on similarities with other titles. By offering tailored recommendations, the system aimed to enhance user satisfaction and reduce churn.

In summary, the project successfully leveraged NLP techniques to analyze Netflix's dataset of TV shows and movies. By clustering content and implementing a content-based recommendation system, the project contributed to improving the user experience and supporting Netflix's efforts to retain subscribers. These findings can aid Netflix in maintaining its competitive edge in the ever-evolving streaming industry.
