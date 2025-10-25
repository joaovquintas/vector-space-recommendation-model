# Vector-Based Movie Recommendation System

A **content-based recommendation system** that uses vector similarity between movie metadata (overview, genres, keywords, cast, and crew) to suggest similar titles.

---

## Objective

Learn and implement the fundamentals of **Recommendation Systems** and **Text Vectorization** by comparing movies based on their textual attributes using cosine similarity.

---

## Project Pipeline

1. **Load the data** (`dataset_filmes.csv` and `dataset_elenco.csv`)  
2. **Data cleaning & preprocessing** — handle missing values, parse JSON-like fields using `ast.literal_eval`  
3. **Feature engineering** — extract relevant columns (genres, cast, crew, keywords, overview)  
4. **Text representation** using `CountVectorizer`  
5. **Cosine similarity computation** between all movie vectors  
6. **Recommendation function** that retrieves the top K most similar movies  
7. **Evaluation** using the *Mean Recommendation Similarity* metric  

---

## Evaluation Metric — *Mean Recommendation Similarity (MRS)*

### Definition

The **Mean Recommendation Similarity (MRS)** quantifies how similar the recommended items are to the base item using cosine similarity.

**Mathematically:**

MRS = (1 / N) * Σᵢ₌₁ᴺ (1 / k) * Σⱼ₌₁ᵏ cosine(vᵢ, vⱼ)

Where:

- **N** = total number of items evaluated  
- **k** = number of recommendations per item  
- **vᵢ** = vector representation of the base movie  
- **vⱼ** = vector representation of each recommended movie  

### Why this metric

- **Unsupervised evaluation:** Works without explicit user feedback (e.g., no ratings dataset required).  
- **Model coherence:** Measures internal consistency — whether the model recommends items that are semantically close to the base item.  
- **Sensitivity to text representation:** Clearly reflects improvements in vectorization methods (e.g., CountVectorizer → TF-IDF → Embeddings).  

### Interpretation of results

For the current version of the model:  
> **MRS (k=5) = 0.2848** → *Low similarity*  

This indicates that the model is **partially identifying relationships** between movies but still relies on **shallow lexical overlaps** rather than deeper semantic understanding.  

Possible causes:
- Simple representation of texts — if you used CountVectorizer, it only counts words, without considering importance or context.
- -Lots of noisy fields (e.g. generic words, common names, synonyms).
-Little balance between characteristics — perhaps the “overview” (synopsis) has much more weight than genres or cast.

---

## Technologies Used

- Python 3  
- Pandas, NumPy  
- Scikit-learn  
- NLTK (for tokenization and stemming)

---

## Author

Developed as part of a learning project on **Machine Learning** and **Content-Based Recommendation Systems**.

---

## References
- Data Science Academy - Machine Learning Course
