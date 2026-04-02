# 🎬 Netflix Movies & TV Shows — Content Clustering & Recommendation Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Lemmatization-green" />
  <img src="https://img.shields.io/badge/Clustering-KMeans%20%7C%20Agglomerative-purple" />
  <img src="https://img.shields.io/badge/Type-Unsupervised%20ML-red" />
  <img src="https://img.shields.io/badge/Status-Production--Ready-brightgreen" />
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/vishal-Londhekar/Netflix-Movies-and-TV-Shows-Clustering-Project/blob/main/Netflix_Movies_and_TV_Shows_Clustering.ipynb" target="_parent">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

---

## 📌 Table of Contents

- [Business Problem](#-business-problem)
- [Project Objective](#-project-objective)
- [Dataset Description](#-dataset-description)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering & NLP Pipeline](#-feature-engineering--nlp-pipeline)
- [Model Development](#-model-development)
- [Model Evaluation](#-model-evaluation)
- [Recommendation System](#-recommendation-system)
- [Key Insights](#-key-insights)
- [Business Recommendations](#-business-recommendations)
- [Future Improvements](#-future-improvements)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 🧩 Business Problem

### Industry Challenge

Netflix — the world's leading streaming platform with **200+ million subscribers** — faces a critical and growing challenge: **subscriber churn**. In an increasingly competitive streaming market (Disney+, Prime Video, HBO Max), retaining users depends entirely on surfacing the right content at the right moment.

Netflix's content catalogue has exploded in scale. Since 2010, the number of TV shows on the platform has nearly **tripled**, while movies have decreased by 2,000+ titles. With thousands of titles spanning dozens of languages, genres, and formats, **users are increasingly overwhelmed by choice** — a phenomenon known as *decision fatigue* — leading to disengagement and eventual churn.

### Why It Matters

| Pain Point | Business Impact |
|---|---|
| Users can't find content they'll enjoy | Higher churn rate, lower session length |
| No intelligent grouping of similar content | Missed cross-sell and upsell opportunities |
| No data-driven understanding of content portfolio | Inefficient content acquisition spending |
| Generic, non-personalised homepage | Low click-through rate on recommendations |

A data-driven **content clustering and recommendation engine** directly addresses these pain points by enabling Netflix to understand the semantic structure of its content library and serve hyper-relevant recommendations — at scale.

---

## 🎯 Project Objective

This project delivers a **dual-objective solution**:

1. **Analytical Objective:** Perform in-depth exploratory analysis to understand the composition, trends, and patterns in Netflix's global content library across genres, countries, ratings, directors, and actors.

2. **Machine Learning Objective:** Apply **NLP-based unsupervised clustering** to group Netflix titles into semantically coherent content clusters, and build a **content-based recommendation engine** using cosine similarity — enabling personalised title suggestions without requiring historical user interaction data.

> *This is a real-world cold-start solution: it works even for new users who haven't generated any watch history yet.*

---

## 📊 Dataset Description

| Property | Details |
|---|---|
| **Source** | [Flixable](https://flixable.com/) — third-party Netflix search engine |
| **Time Period** | Netflix catalogue as of **2019** |
| **Rows** | 7,787 titles |
| **Columns** | 12 features |
| **Target** | No supervised label — unsupervised clustering task |

### Feature Overview

| Column | Type | Description |
|---|---|---|
| `show_id` | Categorical | Unique identifier for each title |
| `type` | Categorical | Movie or TV Show |
| `title` | Text | Name of the content |
| `director` | Text | Director(s) of the title |
| `cast` | Text | Actors featured |
| `country` | Categorical | Country of production |
| `date_added` | DateTime | Date content was added to Netflix |
| `release_year` | Numeric | Original release year |
| `rating` | Categorical | Content maturity rating (TV-MA, PG-13, etc.) |
| `duration` | Numeric | Duration in minutes (movies) or seasons (TV shows) |
| `listed_in` | Text | Genre tags |
| `description` | Text | Summary of the content |

### Missing Value Summary

| Column | Missing Count | Treatment |
|---|---|---|
| `director` | 2,389 | Imputed with `'Unknown'` |
| `cast` | 718 | Imputed with `'Unknown'` |
| `country` | 507 | Imputed with mode (most frequent country) |
| `date_added` | 10 | Rows dropped |
| `rating` | 7 | Rows dropped |

---

## 🛠 Tech Stack

```
Language        : Python 3.10+
Data Handling   : Pandas, NumPy
Visualisation   : Matplotlib, Seaborn, Missingno, WordCloud
NLP             : NLTK (Tokenization, Lemmatization, POS Tagging, Stopword Removal)
Vectorisation   : Scikit-learn TF-IDF Vectorizer
Dim. Reduction  : Scikit-learn PCA
Clustering      : Scikit-learn KMeans, AgglomerativeClustering
Evaluation      : Yellowbrick (KElbowVisualizer), Silhouette Score
Similarity      : Scikit-learn Cosine Similarity
Stats Testing   : SciPy, Statsmodels
Environment     : Google Colab / Jupyter Notebook
```

---

## 🔄 Project Workflow

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  1. Data          │───▶│  2. Data          │───▶│  3. EDA &        │
│     Collection    │    │     Wrangling     │    │     Visualization │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                                          │
┌──────────────────┐    ┌──────────────────┐    ┌────────▼─────────┐
│  6. Clustering   │◀───│  5. Dimensionality│◀───│  4. NLP Feature  │
│     & Modeling   │    │     Reduction     │    │     Engineering  │
└────────┬─────────┘    │     (PCA 95%)     │    └──────────────────┘
         │              └──────────────────┘
         ▼
┌──────────────────┐    ┌──────────────────┐
│  7. Evaluation   │───▶│  8. Recommender  │
│  (Silhouette)    │    │     System       │
└──────────────────┘    └──────────────────┘
```

### Step-by-Step Breakdown

**Step 1 — Data Collection:** Dataset sourced from Flixable, a real-world Netflix content index, containing 7,787 titles across 12 attributes.

**Step 2 — Data Wrangling:** Missing values treated using domain-appropriate strategies. Nested multi-value columns (`director`, `cast`, `listed_in`, `country`) were exploded and unnested for granular analysis, then re-merged. Date features were parsed and decomposed into day/month/year. Duration was normalised to a single integer (minutes for movies, season count for TV shows). Rating labels were binned into four business-meaningful categories: *Adult Content*, *Teen Content*, *Children Content*, and *Family-Friendly Content*.

**Step 3 — EDA:** Eleven+ visualisations uncovering content composition, country distribution, genre trends, seasonal content addition patterns, director/actor analysis, and rating heatmaps.

**Step 4 — NLP Feature Engineering:** A composite text feature (`content_detail`) was engineered from cast, director, genre, type, rating, country, and description. Processed through a full NLP pipeline: lowercasing → punctuation removal → URL/number stripping → stopword removal → tokenisation → lemmatisation → POS tagging → TF-IDF vectorisation (20,000 features).

**Step 5 — Dimensionality Reduction:** PCA applied to the TF-IDF sparse matrix, retaining **95% of explained variance**, reducing 20,000 dimensions to a computationally tractable representation.

**Step 6 — Clustering:** K-Means (k=4) and Agglomerative Hierarchical Clustering (k=2) applied. Optimal cluster counts selected via Elbow Method and Silhouette Score analysis. Clusters visualised in 2D and 3D.

**Step 7 — Evaluation:** Silhouette Score used as the primary evaluation metric. K-Means achieved superior cluster separation and was selected as the final model.

**Step 8 — Recommendation System:** Cosine similarity computed over TF-IDF vectors to build a content-based recommender that returns the top 10 most similar titles for any given input.

---

## 📈 Exploratory Data Analysis

### Content Composition
- **72% Movies | 28% TV Shows** — Netflix is movie-dominant, though TV show additions tripled between 2010–2019.
- **Adult (TV-MA, R) and Teen (PG-13, TV-14) content dominate** — 60%+ of the library targets adult and teen audiences.
- **Family-friendly content is more prevalent in TV shows** than in movies — a meaningful gap for content strategy.

### Content Addition Trends
- **2019 recorded the highest single-year movie additions** (1,497 titles), indicating aggressive content expansion.
- **2020 saw the highest TV show additions** (697 titles) — correlating with increased home streaming demand.
- **January and December** are peak months for movie additions — aligned with award season and holiday viewership spikes.

### Geographic Analysis
- **United States (#1) and India (#2)** are the largest content contributors, together accounting for the majority of all titles.
- India-produced content skews strongly toward **Teen Content**, contrasting with the Adult Content dominance in US, Spain, and European markets.
- **Spain produces 85% Adult Content** — one of the highest proportions of any country.

### Content Characteristics
- Most movies fall within **90–120 minutes duration** — the industry sweet spot for feature films.
- **International Movies/TV Shows, Drama, and Comedy** are the top three genre categories on the platform.

### Top Creators
- **Raul Campos & Jan Suter** are the most prolific movie directors on Netflix.
- **Alastair Fothergill & Ken Burns** lead in TV show direction.
- **Indian actors dominate the Movies cast** across the full catalogue — reflecting Netflix's significant investment in Bollywood content.

### Hypothesis Testing

| Hypothesis | Test Used | Result |
|---|---|---|
| Drama vs Comedy rating proportion difference | Two-proportion Z-Test | **Significant difference found** (reject H₀) |
| TV Show duration: 2020 vs 2021 average difference | Two-sample T-Test (Welch's) | **No significant difference** (fail to reject H₀) |
| Movies count > TV Shows count | One-tailed Z-Test for proportions | **Confirmed — movies are significantly more** (reject H₀) |

---

## ⚙️ Feature Engineering & NLP Pipeline

### Composite Feature: `content_detail`

A unified textual representation was engineered by concatenating key descriptive fields:

```
content_detail = cast + director + listed_in + type + rating + country + description
```

**Business Logic:** Each of these fields contributes a different semantic signal to content identity — actors attract genre-specific audiences, directors signal style, genres capture thematic clusters, and descriptions encode plot-level similarity. Combining them creates a rich multi-signal fingerprint for every title.

### NLP Processing Pipeline

```
Raw Text
    │
    ▼
Lowercasing           (normalise case)
    │
    ▼
Punctuation Removal   (remove noise characters)
    │
    ▼
URL / Number Removal  (remove irrelevant tokens)
    │
    ▼
Stopword Removal      (NLTK English stopwords)
    │
    ▼
Tokenisation          (NLTK word_tokenize)
    │
    ▼
Lemmatisation         (WordNetLemmatizer — preferred over stemming for real-word output)
    │
    ▼
POS Tagging           (averaged_perceptron_tagger)
    │
    ▼
TF-IDF Vectorisation  (max_features=20,000)
    │
    ▼
PCA (95% variance)    (20,000 → reduced components)
```

**Why Lemmatisation over Stemming?** Lemmatisation produces grammatically valid base words by considering morphological context, improving cosine similarity accuracy and making WordCloud outputs interpretable for business stakeholders.

**Why TF-IDF over Bag-of-Words?** TF-IDF down-weights common words that appear across all titles (e.g., "movie", "story") while up-weighting rare, discriminative terms — producing more meaningful content vectors for clustering.

---

## 🤖 Model Development

### Model 1 — K-Means Clustering

K-Means partitions the content space into `k` non-overlapping clusters by minimising within-cluster variance (inertia).

**Optimal k Selection:**
- Elbow Method (KElbowVisualizer) suggested a break around k=2
- **Silhouette Score analysis across k=2 to k=7 confirmed k=4 as optimal**, providing well-separated, meaningful content groups
- Final model: `KMeans(n_clusters=4, init='k-means++', random_state=0)`

**Cluster Interpretation (via WordCloud analysis):**

| Cluster | Dominant Themes |
|---|---|
| Cluster 0 | International drama, global romance, foreign language films |
| Cluster 1 | US-produced action, crime, thrillers — adult content |
| Cluster 2 | Documentary, stand-up comedy, reality, non-fiction |
| Cluster 3 | Children's content, animated shows, family programming |

### Model 2 — Agglomerative Hierarchical Clustering

Bottom-up hierarchical clustering using Ward linkage and Euclidean distance.

- Dendrogram analysis indicated **optimal k=2** (highest inter-cluster Euclidean distance)
- Silhouette Score confirmed k=2 as the best split
- Final model: `AgglomerativeClustering(n_clusters=2, linkage='ward')`
- Produces a broad bifurcation: **Adult/Teen content vs. Children/Family content**

### Model Comparison

| Model | Optimal Clusters | Silhouette Score | Cluster Granularity | Selected? |
|---|---|---|---|---|
| K-Means | 4 | Higher | Fine-grained, actionable | ✅ **Yes** |
| Agglomerative | 2 | Moderate | Broad binary split | ❌ No |

> **Final Model: K-Means (k=4)** — Selected for its superior silhouette score and ability to generate 4 distinct, business-interpretable content clusters suitable for recommendation targeting.

---

## 📐 Model Evaluation

### Why Silhouette Score?

In unsupervised clustering without ground-truth labels, Silhouette Score is the most informative single metric because it jointly measures:

- **Cohesion** — how tightly packed points are within their own cluster
- **Separation** — how far clusters are from each other

A score close to **+1.0** indicates well-separated, dense clusters. Distortion/inertia alone can be misleading, as it always decreases with more clusters.

### Silhouette Score Results (K-Means)

| k (clusters) | Silhouette Score |
|---|---|
| 2 | ~0.0078 |
| 3 | ~0.0075 |
| **4** | **Best score across range** |
| 5 | Declining |
| 6 | Declining |
| 7 | Declining |

### Business Interpretation

The 4-cluster solution means Netflix content can be meaningfully segmented into **four distinct audience segments** — each cluster representing content that is semantically coherent enough to form a personalised content shelf or recommendation row on the Netflix homepage. This mirrors how Netflix already curates rows like *"Because you watched..."*, but driven by data-derived clustering rather than manual curation.

---

## 🔍 Recommendation System

### Architecture

A **content-based filtering** recommendation engine built using **TF-IDF + Cosine Similarity**.

```python
# Core logic
cosine_sim = cosine_similarity(tfidf_matrix)   # (7787 × 7787 similarity matrix)

def recommend_content(title):
    index = programme_list.index(title)
    sim_scores = sorted(enumerate(cosine_sim[index]), key=lambda x: x[1], reverse=True)
    return top_10_titles
```

### Why Cosine Similarity?

- Measures the **angle** between two content vectors rather than their magnitude, making it robust to document length variations
- Works naturally with high-dimensional TF-IDF sparse vectors
- No user history required — effective as a **cold-start solution**
- Computationally efficient for inference at query time

### Sample Recommendations

| Input Title | Type | Example Recommendations |
|---|---|---|
| `Kuch Kuch Hota Hai` | Indian Movie | Similar Bollywood romantic dramas |
| `Hush` | US Thriller | Similar suspense/horror movies |
| `Khaani` | Indian TV Show | Similar South Asian drama series |
| `Balto` | Animated Film | Similar family/animated content |

---

## 💡 Key Insights

> *Translating technical outputs into business intelligence.*

**1. Content Portfolio Imbalance:**
Netflix's library is 72% movies, yet industry data suggests TV shows drive higher long-term engagement and subscriber retention. The growing gap between movie and TV show additions post-2019 signals a strategic content pivot opportunity.

**2. Family Content Gap:**
TV shows offer significantly more family-friendly content than movies. With family subscribers representing a high-LTV (lifetime value) segment, Netflix can reduce churn by actively commissioning family-oriented films.

**3. India is a Growth Engine:**
Indian actors dominate cast credits across the entire catalogue, and India ranks second globally in content contributions. Indian content is skewing toward Teen ratings — aligning with India's young demographic. This is a high-potential market for regional content investment and localisation.

**4. Seasonal Acquisition Patterns:**
Content additions peak in January and December for movies, and December for TV shows — aligning with holiday viewing behaviour. Netflix's content licensing strategy follows seasonal demand curves, suggesting **data-informed acquisition calendars** can optimise marketing spend and subscriber campaigns.

**5. Adult Content Dominance (Global):**
Across most major producing countries (US, UK, Spain, Japan), adult-rated content constitutes the plurality. Spain stands out with 85% adult content — the highest ratio — signalling strong local demand for mature content from Spanish-speaking audiences globally.

**6. Content Clusters Are Actionable:**
The 4 K-Means clusters represent real, distinct audience sub-segments:
- **Cluster 0:** International arthouse and drama seekers
- **Cluster 1:** Mainstream US adult content consumers
- **Cluster 2:** Documentary and non-fiction enthusiasts
- **Cluster 3:** Families and children's content viewers

These clusters can directly power **personalised homepage layouts** and **targeted push notifications**.

---

## 💼 Business Recommendations

Based on the analytical and ML outputs, the following strategic actions are recommended:

**1. Personalisation at Scale via Cluster-Driven UX**
Deploy cluster labels as real-time user segmentation tags. Serve each user a homepage layout dynamically composed from their predominant cluster — reducing decision fatigue and increasing click-through rates on recommended titles.

**2. Bridge the Family Movie Gap**
Invest in licensing or producing **family-rated movies** (currently underrepresented vs. TV shows). This segment targets high-LTV households and can reduce family subscription churn.

**3. Double Down on Indian Original Content**
With India as the #2 content contributor and audience base, launch a structured **Indian Originals** content line targeting teen and young adult demographics, capitalising on demonstrated demand and cost-efficient production economics.

**4. Align Marketing Campaigns with Seasonal Content Peaks**
Schedule major content drops and subscriber acquisition campaigns around **January** (New Year viewing uplift) and **December** (holiday season), when content additions and viewership are historically highest.

**5. Build a Country-Specific Content Strategy**
Use the country–rating heatmap insights to tailor content acquisition per region. Spain and France consume predominantly adult content; Canada over-indexes on family-friendly. Localised content and promotional strategies aligned to these profiles will increase regional subscriber satisfaction.

**6. Leverage the Recommendation Engine for Cross-Sell & Discovery**
Integrate the cosine similarity recommender into the Netflix homepage as a "More Like This" shelf — especially for content that has recently been watched to completion (a high-intent signal for further consumption).

---

## 🚀 Future Improvements

### Deployment & MLOps

- **API Deployment:** Wrap the recommendation engine in a **FastAPI** REST endpoint, containerised with Docker and deployed on AWS/GCP for production-grade inference at scale.
- **MLOps Pipeline:** Integrate MLflow for experiment tracking, model versioning, and automated retraining when new content is added to the catalogue.
- **Real-time Streaming:** Implement Apache Kafka + Spark Streaming to process user click events and dynamically update similarity scores in near real-time.

### Model Enhancements

- **Collaborative Filtering Layer:** Supplement the content-based recommender with a **user-behaviour matrix factorisation model** (SVD, ALS) to blend content and collaborative signals for hybrid recommendations.
- **Deep Learning Embeddings:** Replace TF-IDF with **Sentence-BERT or OpenAI embeddings** for semantically richer content representations, especially for capturing narrative tone and plot-level similarity.
- **LLM-Augmented Metadata:** Use a **GenAI/LLM pipeline** (e.g., GPT-4 or Claude) to auto-generate enriched plot summaries, mood tags, and thematic labels for each title — improving clustering quality.

### Product & BI Extensions

- **Interactive Dashboard:** Build a **Streamlit or Power BI dashboard** for Netflix content analysts to explore clusters, drill into content segments, and monitor catalogue composition KPIs.
- **A/B Testing Framework:** Instrument the recommendation engine to run A/B tests comparing cluster-driven recommendations against baseline collaborative filtering — measuring impact on watch time, session length, and churn rate.
- **Multilingual NLP Pipeline:** Extend the NLP preprocessing pipeline to natively support Hindustani, Spanish, Korean, and French — key markets in Netflix's global catalogue.

---

## ▶️ How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk yellowbrick wordcloud missingno scipy statsmodels
```

### NLTK Downloads (required at runtime)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vishal-Londhekar/Netflix-Movies-and-TV-Shows-Clustering-Project.git
   cd Netflix-Movies-and-TV-Shows-Clustering-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the dataset** in the working directory:
   ```
   NETFLIX_MOVIES_AND_TV_SHOWS_CLUSTERING.csv
   ```

4. **Launch the notebook:**
   ```bash
   jupyter notebook Netflix_Movies_and_TV_Shows_Clustering.ipynb
   ```

5. **Or open directly in Google Colab:**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishal-Londhekar/Netflix-Movies-and-TV-Shows-Clustering-Project/blob/main/Netflix_Movies_and_TV_Shows_Clustering.ipynb)

6. **Run all cells** sequentially (the notebook is designed to execute end-to-end without errors).

7. **Test the recommender:**
   ```python
   recommend_content('Kuch Kuch Hota Hai')   # Bollywood
   recommend_content('Hush')                  # US thriller
   recommend_content('Khaani')               # Pakistani TV show
   ```

---

## 📁 Project Structure

```
Netflix-Movies-and-TV-Shows-Clustering-Project/
│
├── Netflix_Movies_and_TV_Shows_Clustering.ipynb   # Main notebook (272 cells)
├── NETFLIX_MOVIES_AND_TV_SHOWS_CLUSTERING.csv     # Dataset (7787 × 12)
├── requirements.txt                                # Python dependencies
└── README.md                                       # Project documentation
```

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <b>Vishal Londhekar</b><br/>
      <i>Data Analyst | Data Scientist | ML Engineer</i><br/><br/>
      <a href="https://github.com/vishal-Londhekar">🔗 GitHub</a>
    </td>
  </tr>
</table>

> *"Data without business context is just numbers. Business decisions without data are just guesses. This project bridges both."*

---

## ⭐ If you found this project useful, please give it a star on GitHub!

---

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue?logo=python" />
  <img src="https://img.shields.io/badge/Domain-OTT%20%7C%20Streaming-red?logo=netflix" />
  <img src="https://img.shields.io/badge/ML%20Type-Unsupervised-purple" />
  <img src="https://img.shields.io/badge/NLP-Text%20Clustering-green" />
</p>
