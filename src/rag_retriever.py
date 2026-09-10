import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HistoricalRAGRetriever:
    """
    Retrieval-Augmented Generation (RAG) Grounding Index.
    Indexes past @AmazonHelp customer-brand interactions and retrieves top historical
    resolutions matching customer queries to ground agent replies.
    """
    def __init__(self, parquet_path: str = "data/amazon_help_pairs.parquet"):
        self.parquet_path = parquet_path
        self.df = None
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
        self.tfidf_matrix = None
        self._is_indexed = False

    def load_and_index(self):
        """Loads AmazonHelp pairs and builds TF-IDF vector index."""
        if self._is_indexed:
            return

        self.df = pd.read_parquet(self.parquet_path)
        # Limit index size for fast execution
        if len(self.df) > 5000:
            self.df = self.df.sample(5000, random_state=42).reset_index(drop=True)

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['customer_text'].fillna(''))
        self._is_indexed = True
        print(f"RAG Retriever indexed {len(self.df)} historical resolution pairs.")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top_k most relevant past customer-brand resolution pairs."""
        if not self._is_indexed:
            self.load_and_index()

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            row = self.df.iloc[idx]
            results.append({
                "historical_customer_query": row['customer_text'],
                "historical_brand_reply": row['brand_text'],
                "similarity_score": round(score, 4)
            })

        return results
