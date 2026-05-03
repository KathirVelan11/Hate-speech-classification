import pandas as pd
import numpy as np
import pickle
import warnings
import os
import torch
import argparse
import faiss
import gc
import psutil
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

def print_memory_usage(label=""):
    prefix = f"[{label}] " if label else ""
    process = psutil.Process()
    ram_used = process.memory_info().rss / 1024 ** 3
    ram_avail = psutil.virtual_memory().available / 1024 ** 3
    
    print(f"\n{prefix}📊 Memory Usage:")
    print(f"  RAM Used  : {ram_used:.2f} GB")
    print(f"  RAM Avail : {ram_avail:.2f} GB")
    
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / 1024 ** 3
        gpu_reserv = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"  GPU Alloc : {gpu_alloc:.2f} GB")
        print(f"  GPU Rsvrd : {gpu_reserv:.2f} GB")

print_memory_usage("START")

# ============================================================================
# INFERENCE SCRIPT FOR EXPLAINABLE HATE SPEECH CLASSIFICATION
# ============================================================================

print("=" * 80)
print("EXPLAINABLE HATE SPEECH CLASSIFICATION - INFERENCE")
print("=" * 80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run inference on text for hate speech classification')
parser.add_argument('--text', type=str, help='Input text to classify (if not provided, will prompt)')
parser.add_argument('--top_k', type=int, default=5, help='Number of top phrases to extract (default: 5)')


args = parser.parse_args()

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

print("\n[1/3] Loading models and data...")

# Check GPU availability
gpu_available = torch.cuda.is_available()
device = 'cuda' if gpu_available else 'cpu'
print(f"GPU Available: {gpu_available} (using device: {device})")

# Load fine-tuned embedding model (with fallback to base model)
try:
    embedding_model_path = 'hate_speech_fine_tuned_model'
    if os.path.exists(embedding_model_path):
        embedding_model = SentenceTransformer(embedding_model_path, device=device)
        print(f"[OK] Loaded fine-tuned embedding model from {embedding_model_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"[FALLBACK] Fine-tuned model not found at {embedding_model_path}")
        print("[FALLBACK] Using base model: paraphrase-mpnet-base-v2")
        embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)
        print(f"[OK] Loaded base embedding model")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
except Exception as e:
    print(f"ERROR: Failed to load embedding model: {e}")
    raise

# Initialize KeyBERT with fine-tuned model
try:
    kw_model = KeyBERT(model=embedding_model)
    print("✓ Initialized KeyBERT model")
except Exception as e:
    print(f"ERROR: Failed to initialize KeyBERT: {e}")
    raise

# Load custom stop words
try:
    with open('custom_stop_words.pkl', 'rb') as f:
        custom_stop_words = pickle.load(f)
    print("✓ Loaded custom stop words")
except Exception as e:
    print(f"ERROR: Failed to load custom stop words: {e}")
    custom_stop_words = set()  # Fallback

# Load unique phrases and embeddings
try:
    with open('unique_phrases.pkl', 'rb') as f:
        unique_phrases = pickle.load(f)
    
    # Load embeddings as memmap (1.77M phrases x 768 dimensions)
    phrase_embeddings = np.memmap('phrase_embeddings.npy', dtype='float32', mode='r', shape=(1776478, 768))
    print(f"✓ Loaded {len(unique_phrases)} unique phrases with embeddings")
    
    # Build FAISS index for faster similarity search
    index = faiss.IndexFlatIP(768)
    index.add(np.ascontiguousarray(phrase_embeddings).astype('float32'))
    
    print(f"✓ Built FAISS index for {len(unique_phrases)} phrases")
except Exception as e:
    print(f"ERROR: Failed to load unique phrases and embeddings: {e}")
    raise

cluster_model_path = 'kmeans_clustering.pkl'
phrases_csv = 'phrases_with_clusters.csv'
stats_csv = 'cluster_statistics.csv'

try:
    with open(cluster_model_path, 'rb') as f:
        clustering_model = pickle.load(f)
    print(f"✓ Loaded clustering model from {cluster_model_path}")
except Exception as e:
    print(f"ERROR: Failed to load clustering model from {cluster_model_path}")
    print("Please run clustering.py first to train the clustering model.")
    raise

try:
    cluster_stats_df = pd.read_csv(stats_csv)
    # Handle both old 'hate_ratio' and new 'cluster_hate_ratio' column names
    hate_ratio_col = 'cluster_hate_ratio' if 'cluster_hate_ratio' in cluster_stats_df.columns else 'hate_ratio'
    # Remove '%' sign if present and convert to float
    cluster_dict = {}
    for idx, row in cluster_stats_df.iterrows():
        ratio_str = str(row[hate_ratio_col]).replace('%', '').strip()
        cluster_dict[row['cluster_id']] = float(ratio_str) / 100.0
    print(f"[OK] Loaded cluster statistics from {stats_csv}")
except Exception as e:
    print(f"ERROR: Failed to load cluster statistics from {stats_csv}")
    print(f"Available columns: {cluster_stats_df.columns.tolist() if 'cluster_stats_df' in locals() else 'N/A'}")
    raise

try:
    phrases_df = pd.read_csv(phrases_csv, low_memory=False)
    print(f"✓ Loaded phrases with clusters from {phrases_csv}")
except Exception as e:
    print(f"ERROR: Failed to load phrases with clusters from {phrases_csv}")
    raise

phrase_to_cluster = dict(zip(phrases_df['phrase'], phrases_df['cluster']))

gc.collect()

print(f"\n📊 Model loaded successfully!")
print(f"  - Total clusters: {len(cluster_dict)}")
print(f"  - Total phrases: {len(phrases_df)}")


# ============================================================================
# PHRASE EXTRACTION FUNCTION
# ============================================================================

@lru_cache(maxsize=1000)
def extract_key_phrases(text, top_k=5):
    """Extract top k key phrases from text using KeyBERT with syntactic filtering."""
    if not text.strip():
        return []

    # Extract keywords using KeyBERT
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 4),
        stop_words=list(custom_stop_words),
        top_n=top_k * 2,  # Extract more to filter
        use_mmr=True,
        diversity=0.7
    )

    # Filter and clean phrases
    filtered_phrases = []
    for phrase, score in keywords:
        # Skip if in custom stop words
        if phrase.lower() in custom_stop_words:
            continue

        # Add phrase without syntactic filtering (spaCy removed)
        filtered_phrases.append((phrase, score))

    # Return top k phrases
    return filtered_phrases[:top_k]


# ============================================================================
# CLASSIFICATION FUNCTION
# ============================================================================

def classify_phrase(phrase, embedding_model, clustering_model, unique_phrases, index, phrase_to_cluster, cluster_dict, cluster_hate_lookup=None):
    """Classify a phrase using K-nearest neighbors' cluster hate ratios."""

    # Compute embedding for the phrase
    phrase_embedding = embedding_model.encode([phrase], convert_to_numpy=True, normalize_embeddings=True)

    # Find K=5 nearest neighbors using FAISS
    K = 5
    similarities, indices = index.search(phrase_embedding.astype('float32'), K)
    
    low_confidence = similarities[0][0] < 0.6  # Confidence based on closest neighbor
    
    # Get hate ratios from K nearest neighbors' clusters
    hate_ratios = []
    cluster_ids = []
    
    for idx in indices[0]:
        closest_phrase = unique_phrases[idx]
        cluster_id = phrase_to_cluster.get(closest_phrase, -1)
        cluster_ids.append(cluster_id)
        
        if cluster_id >= 0 and cluster_id in cluster_dict:
            hate_ratio = cluster_dict[cluster_id]
            hate_ratios.append(hate_ratio)
    
    if not hate_ratios:
        return {
            'phrase': phrase,
            'cluster_id': 'Unknown',
            'hate_ratio': 'N/A',
            'classification': 'Unable to classify',
            'confidence': similarities[0][0],
            'low_confidence': low_confidence,
            'classification_method': 'N/A'
        }
    
    # Use mean hate ratio of K neighbors
    mean_hate_ratio = np.mean(hate_ratios)
    primary_cluster = cluster_ids[0] if cluster_ids else 'Unknown'
    
    # Threshold: 0.50 for hate speech classification
    classification = "Hate speech" if mean_hate_ratio > 0.50 else "Non-hate speech"
    
    return {
        'phrase': phrase,
        'cluster_id': primary_cluster,
        'hate_ratio': f"{mean_hate_ratio:.3f}",
        'classification': classification,
        'confidence': similarities[0][0],
        'low_confidence': low_confidence,
        'classification_method': f'KNN-based ({K} neighbors)'
    }


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def run_inference(text, top_k=5):
    """Run inference on input text and return explainable results."""

    print(f"\n🔍 Analyzing text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")

    # Extract key phrases for explainability
    print(f"\n📝 Extracting top {top_k} key phrases...")
    extracted_phrases = extract_key_phrases(text, top_k=top_k)

    if not extracted_phrases:
        print("❌ No key phrases extracted from the text.")
        return {
            'overall_classification': 'Unable to classify',
            'phrases': [],
            'summary': 'No phrases could be extracted for analysis.'
        }

    print(f"✓ Extracted {len(extracted_phrases)} phrases:")
    for i, (phrase, score) in enumerate(extracted_phrases, 1):
        print(f"  {i}. \"{phrase}\" (score: {score:.3f})")

    # =====================================================================
    # PRIMARY METHOD: Classify based on input text embedding similarity
    # =====================================================================
    print(f"\n🤖 Classifying text based on embedding similarity...")
    
    # Get embedding of entire input text
    text_embedding = embedding_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    
    # Find K=10 nearest phrases to the input text
    K_NEIGHBORS = 10
    similarities, indices = index.search(text_embedding.astype('float32'), K_NEIGHBORS)
    
    # Get hate ratios of nearest neighbors' clusters
    neighbor_hate_ratios = []
    for idx in indices[0]:
        closest_phrase = unique_phrases[idx]
        cluster_id = phrase_to_cluster.get(closest_phrase, -1)
        if cluster_id >= 0 and cluster_id in cluster_dict:
            hate_ratio = cluster_dict[cluster_id]
            neighbor_hate_ratios.append(hate_ratio)
    
    if neighbor_hate_ratios:
        avg_neighbor_hate_ratio = np.mean(neighbor_hate_ratios)
        text_similarity_confidence = similarities[0][0]
        
        # Adaptive threshold based on similarity confidence
        # Higher confidence allows LOWER hate ratio threshold (hate is clearer when phrases match well)
        # Lower confidence requires HIGHER hate ratio threshold (conservative when similarity is low)
        if text_similarity_confidence < 0.5:
            threshold = 0.85  # Very low confidence: require 85% hate ratio
        elif text_similarity_confidence < 0.65:
            threshold = 0.80  # Low confidence: require 80% hate ratio
        elif text_similarity_confidence < 0.80:
            threshold = 0.75  # Medium confidence: require 75% hate ratio
        elif text_similarity_confidence < 0.90:
            threshold = 0.70  # High confidence: require 70% hate ratio
        else:
            threshold = 0.65  # Very high confidence: require 65% hate ratio
        
        text_classification = "Hate speech detected" if avg_neighbor_hate_ratio > threshold else "Non-hate speech"
        print(f"  Input text similarity to phrases: {text_similarity_confidence:.3f}")
        print(f"  Average hate ratio of nearest {K_NEIGHBORS} phrases: {avg_neighbor_hate_ratio:.3f}")
        print(f"  Adaptive threshold (based on confidence): {threshold:.2f}")
        print(f"  Classification: {text_classification}")
    else:
        avg_neighbor_hate_ratio = 0
        text_similarity_confidence = 0
        text_classification = "Unable to classify"
        threshold = 0.50
    
    # =====================================================================
    # SECONDARY: Extract phrase classifications for explainability
    # =====================================================================
    print(f"\n📋 Extracting phrase-level information for explainability...")
    classifications = []

    for phrase, score in extracted_phrases:
        result = classify_phrase(
            phrase,
            embedding_model,
            clustering_model,
            unique_phrases,
            index,
            phrase_to_cluster,
            cluster_dict
        )
        classifications.append(result)

        print(f"  \"{phrase}\" → Cluster {result['cluster_id']}, "
              f"Hate Ratio: {result['hate_ratio']}, "
              f"Classification: {result['classification']}")

    # Summary
    summary = f"Analysis complete: Input text classified based on semantic similarity to {K_NEIGHBORS} nearest phrases (avg hate ratio: {avg_neighbor_hate_ratio:.3f})"

    return {
        'overall_classification': text_classification,
        'phrases': classifications,
        'summary': summary,
        'text_analysis': {
            'similarity_confidence': text_similarity_confidence,
            'avg_hate_ratio': avg_neighbor_hate_ratio
        }
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Get input text
        if args.text:
            input_text = args.text
        else:
            print("\n💬 Enter the text you want to analyze for hate speech:")
            input_text = input(">>> ")

        if not input_text.strip():
            print("❌ No text provided. Exiting.")
            exit(1)

        # Run inference
        results = run_inference(input_text, top_k=args.top_k)

        # Display results
        print(f"\n" + "=" * 80)
        print("📊 INFERENCE RESULTS")
        print("=" * 80)

        print(f"\n🎯 Overall Classification: {results['overall_classification']}")
        print(f"\n📋 Detailed Analysis:")

        for i, phrase_result in enumerate(results['phrases'], 1):
            print(f"\n  {i}. Phrase: \"{phrase_result['phrase']}\"")
            print(f"     Cluster ID: {phrase_result['cluster_id']}")
            print(f"     Hate Ratio: {phrase_result['hate_ratio']}")
            print(f"     Classification: {phrase_result['classification']}")
            print(f"     Confidence: {phrase_result['confidence']:.3f}{' (LOW CONFIDENCE)' if phrase_result.get('low_confidence') else ''}")

        print(f"\n📝 Summary: {results['summary']}")

        print(f"\n" + "=" * 80)
        print("INFERENCE COMPLETE!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n❌ Inference interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎯 Overall Classification: {results['overall_classification']}")


    print_memory_usage("END")