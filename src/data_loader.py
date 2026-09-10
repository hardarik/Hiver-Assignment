import os
import re
import pandas as pd

def clean_tweet_text(text: str) -> str:
    """Clean tweet text by removing handle tags if necessary while maintaining meaning."""
    if not isinstance(text, str):
        return ""
    # Replace multiple spaces / newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_english(text: str, threshold: float = 0.85) -> bool:
    """Filter out non-English tweets using ASCII ratio heuristic."""
    if not isinstance(text, str) or not text.strip():
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= threshold

def prepare_amazon_help_dataset(csv_path: str = "data/twcs_sample.csv", output_parquet: str = "data/amazon_help_pairs.parquet") -> pd.DataFrame:
    """
    Extracts English Customer -> AmazonHelp pair conversations from twcs_sample.csv
    and saves to parquet for fast retrieval indexing.
    """
    if os.path.exists(output_parquet):
        return pd.read_parquet(output_parquet)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input dataset file not found at {csv_path}")

    print(f"Loading raw TWCS dataset from {csv_path}...")
    df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)

    # Filter AmazonHelp brand replies
    brand_replies = df[df['author_id'] == 'AmazonHelp'].copy()
    brand_replies['in_response_to_tweet_id'] = pd.to_numeric(brand_replies['in_response_to_tweet_id'], errors='coerce')
    brand_replies = brand_replies.dropna(subset=['in_response_to_tweet_id'])

    # Filter customer inbound tweets
    customer_tweets = df[df['inbound'] == True].copy()
    customer_tweets['tweet_id'] = pd.to_numeric(customer_tweets['tweet_id'], errors='coerce')

    # Merge customer tweet with AmazonHelp response
    merged = pd.merge(
        customer_tweets,
        brand_replies,
        left_on='tweet_id',
        right_on='in_response_to_tweet_id',
        suffixes=('_cust', '_brand')
    )

    merged['text_cust_clean'] = merged['text_cust'].apply(clean_tweet_text)
    merged['text_brand_clean'] = merged['text_brand'].apply(clean_tweet_text)

    # Filter English
    merged_en = merged[
        merged['text_cust_clean'].apply(is_english) & 
        merged['text_brand_clean'].apply(is_english)
    ].copy()

    # Rename columns for clarity
    output_df = pd.DataFrame({
        'pair_id': merged_en['tweet_id_brand'].astype(str),
        'customer_tweet_id': merged_en['tweet_id_cust'].astype(str),
        'customer_id': merged_en['author_id_cust'].astype(str),
        'customer_text': merged_en['text_cust_clean'],
        'brand_text': merged_en['text_brand_clean'],
        'created_at_cust': merged_en['created_at_cust'],
        'created_at_brand': merged_en['created_at_brand']
    })

    # Deduplicate and reset index
    output_df = output_df.drop_duplicates(subset=['customer_text']).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    output_df.to_parquet(output_parquet, index=False)
    print(f"Saved {len(output_df)} cleaned @AmazonHelp pairs to {output_parquet}")
    return output_df

if __name__ == "__main__":
    df = prepare_amazon_help_dataset()
    print("Dataset shape:", df.shape)
    print(df.head(3))
