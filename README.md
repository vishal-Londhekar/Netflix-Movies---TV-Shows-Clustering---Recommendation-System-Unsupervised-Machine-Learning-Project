# Netflix — Movies & TV Shows Clustering and Recommendation (Unsupervised ML)

## Project overview
This repository contains an unsupervised machine learning project that clusters Netflix titles (movies and TV shows) using genre and metadata to identify content groups and support content-based recommendation ideas. The analysis is implemented in a Jupyter Notebook and uses standard Python data-science libraries.

## Files in this repo
- `NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv` — dataset used for analysis
- `Netflix_Movies_and_TV_Shows_Clustering.ipynb` — main Jupyter Notebook with EDA, preprocessing, clustering, and visualizations
- `README.md` — this file

## Dataset
- Source: Kaggle — Netflix Movies and TV Shows dataset (link in notebook)
- Records: ~6,000+ titles
- Typical fields: `title`, `type` (Movie/TV Show), `listed_in` (genres), `release_year`, `country`, `rating`, etc.

## Key goals
- Preprocess and clean the dataset (handle missing values, parse multi-label genres)
- Represent titles as a genre-feature matrix (one-hot / multi-label encoding)
- Run K-Means clustering to group similar titles
- Reduce dimensions with PCA for visualization
- Interpret clusters and suggest how they could feed a simple content-based recommendation step

## Technologies & libraries
- Python 3.8+
- pandas, numpy
- scikit-learn (KMeans, PCA)
- matplotlib, seaborn
- jupyter / notebook

## Setup & quick start
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install common data-science packages (if you don't have a `requirements.txt`):

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. Launch the notebook and run cells:

```powershell
jupyter notebook "Netflix_Movies_and_TV_Shows_Clustering.ipynb"
```

Open the notebook and run the cells in order. The notebook includes explanations, plots, and the clustering pipeline.

## Methodology (high level)
1. Data cleaning: drop or impute missing values, normalize text fields, parse `listed_in` into individual genres.
2. Feature engineering: create a binary/one-hot genre matrix and optionally include other numeric features (year, rating encoded).
3. Select K with the Elbow method or silhouette analysis.
4. Fit K-Means and analyze cluster centers to label clusters (e.g., Action-focused, Family/Kids, Drama/Romance).
5. Use PCA to visualize clusters in 2D.

## Results & interpretation
- The notebook demonstrates how clusters separate titles by dominant genre combinations and some temporal/country patterns.
- Use cluster membership as a simple content grouping for recommendations (e.g., recommend titles from the same cluster).

## Reproducibility notes
- Ensure the CSV file `NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv` is in the repository root alongside the notebook.
- If random initialization affects clustering, set `random_state` in scikit-learn KMeans for deterministic runs.

## Next steps (optional)
- Create a `requirements.txt` to pin package versions.
- Add a small script to export cluster assignments to CSV for downstream use.
- Experiment with alternative clustering methods (Agglomerative, DBSCAN) or topic modelling on descriptions (if available).

## Author / Contact
Vishal Londhekar — data analysis & clustering. Connect on LinkedIn: https://www.linkedin.com/in/vishal-londhekar

## License
This repository is shared for educational purposes. Add a license file if you want to permit reuse.


















