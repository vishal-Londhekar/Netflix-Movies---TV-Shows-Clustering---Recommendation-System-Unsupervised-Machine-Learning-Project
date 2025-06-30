# 🎬 Netflix Movies and TV Shows Clustering
![images (2)](https://github.com/user-attachments/assets/338d1b55-1880-473d-90fa-ea4af4fa79c7)

## 📌 Objective
To segment Netflix titles using **K-Means Clustering** based on genre and other metadata to uncover viewer patterns and enable content personalization.

## 📂 Dataset
- Source: Kaggle – [Netflix Movies and TV Shows Dataset](https://www.kaggle.com/shivamb/netflix-shows)
- Records: 6,000+ titles
- Features: Title, Genre, Type (Movie/TV Show), Release Year, Country, Rating

## 🧰 Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn (K-Means, PCA)
- Seaborn & Matplotlib for EDA and visualization
- Jupyter Notebook

## 🚀 Process Overview
1. 📊 **Data Cleaning & Preprocessing**
   - Removed nulls, parsed multi-label genres
   - One-hot encoding of genre columns

2. 🔍 **Exploratory Data Analysis (EDA)**
   - Genre distribution, content trends by year and type
   - Visualizations of top genres and countries

3. 📦 **Clustering (K-Means)**
   - Applied K-Means on genre matrix
   - Used Elbow Method to determine optimal number of clusters (K=5)

4. 📉 **Dimensionality Reduction**
   - Applied PCA to visualize content clusters in 2D

## 📈 Key Insights
- Identified 5 distinct viewer/content clusters (e.g., Action, Comedy, Kids, Romance, Thriller)
- Content trends vary by region and type (movies vs shows)
- Useful for marketing segmentation or improving recommendations

## 💡 Business Use Case
Helps platforms like Netflix:
- Understand audience content preferences
- Group titles for personalized recommendation engines
- Plan content acquisition by genre and region


## **Vishal Londhekar**

## **📫 Email: vishal.londhekar1998@gmail.com**
