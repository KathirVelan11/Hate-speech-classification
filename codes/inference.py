import pandas as pd
import numpy as np
import pickle
import warnings
import os
import torch
import argparse
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

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

# Load fine-tuned embedding model
try:
    embedding_model_path = 'hate_speech_fine_tuned_model'
    if os.path.exists(embedding_model_path):
        embedding_model = SentenceTransformer(embedding_model_path, device=device)
        print(f"✓ Loaded fine-tuned embedding model from {embedding_model_path}")
    else:
        print(f"ERROR: Fine-tuned model not found at {embedding_model_path}")
        print("Please run finetuning.py first to train the model.")
        raise FileNotFoundError
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
    phrase_embeddings = np.load('phrase_embeddings.npy')
    print(f"✓ Loaded {len(unique_phrases)} unique phrases and embeddings")
except Exception as e:
    print("ERROR: Failed to load unique phrases and embeddings")
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
    print(f"✓ Loaded cluster statistics from {stats_csv}")
except Exception as e:
    print(f"ERROR: Failed to load cluster statistics from {stats_csv}")
    raise

try:
    phrases_df = pd.read_csv(phrases_csv, low_memory=False)
    print(f"✓ Loaded phrases with clusters from {phrases_csv}")
except Exception as e:
    print(f"ERROR: Failed to load phrases with clusters from {phrases_csv}")
    raise

phrase_to_cluster = dict(zip(phrases_df['phrase'], phrases_df['cluster']))

print(f"\n📊 Model loaded successfully!")
print(f"  - Total clusters: {len(cluster_stats_df)}")
print(f"  - Total phrases: {len(phrases_df)}")


# ============================================================================
# PHRASE EXTRACTION FUNCTION
# ============================================================================

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

def classify_phrase(phrase, embedding_model, clustering_model, unique_phrases, phrase_embeddings, phrase_to_cluster, cluster_stats_df):
    """Classify a phrase by finding its cluster and determining hate/non-hate based on cluster statistics."""

    # Compute embedding for the phrase
    phrase_embedding = embedding_model.encode([phrase], convert_to_numpy=True, normalize_embeddings=True)

    # Find closest cluster using cosine similarity to existing phrases
    similarities = cosine_similarity(phrase_embedding, phrase_embeddings)[0]
    closest_idx = np.argmax(similarities)
    closest_phrase = unique_phrases[closest_idx]
    cluster_id = phrase_to_cluster.get(closest_phrase, -1)

    if cluster_id == -1:
        return {
            'phrase': phrase,
            'cluster_id': 'Unknown',
            'hate_ratio': 'N/A',
            'classification': 'Unable to classify',
            'confidence': similarities[closest_idx]
        }

    # Get cluster statistics
    cluster_info = cluster_stats_df[cluster_stats_df['cluster_id'] == cluster_id]
    if len(cluster_info) == 0:
        return {
            'phrase': phrase,
            'cluster_id': cluster_id,
            'hate_ratio': 'N/A',
            'classification': 'Unable to classify',
            'confidence': similarities[closest_idx]
        }

    hate_ratio = cluster_info['hate_ratio'].values[0]
    classification = "Hate speech" if hate_ratio > 0.5 else "Non-hate speech"

    return {
        'phrase': phrase,
        'cluster_id': cluster_id,
        'hate_ratio': f"{hate_ratio:.3f}",
        'classification': classification,
        'confidence': similarities[closest_idx]
    }


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def run_inference(text, top_k=5):
    """Run inference on input text and return explainable results."""

    print(f"\n🔍 Analyzing text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")

    # Extract key phrases
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

    # Classify each phrase
    print(f"\n🤖 Classifying phrases...")
    classifications = []

    for phrase, score in extracted_phrases:
        result = classify_phrase(
            phrase,
            embedding_model,
            clustering_model,
            unique_phrases,
            phrase_embeddings,
            phrase_to_cluster,
            cluster_stats_df
        )
        classifications.append(result)

        print(f"  \"{phrase}\" → Cluster {result['cluster_id']}, "
              f"Hate Ratio: {result['hate_ratio']}, "
              f"Classification: {result['classification']}")

    # Overall classification based on mean hate ratio
    total_phrases = len(classifications)
    valid_hate_ratios = []

    for result in classifications:
        if result['hate_ratio'] != 'N/A':
            try:
                valid_hate_ratios.append(float(result['hate_ratio']))
            except ValueError:
                pass

    if valid_hate_ratios:
        mean_hate_ratio = np.mean(valid_hate_ratios)
        if mean_hate_ratio > 0.5:
            overall_classification = "Hate speech detected"
        elif mean_hate_ratio == 0.5:
            overall_classification = "Neutral content (balanced hate/non-hate phrases)"
        else:
            overall_classification = "Non-hate speech"
    else:
        overall_classification = "Unable to classify (no valid hate ratios)"
        mean_hate_ratio = None

    # Summary
    if mean_hate_ratio is not None:
        summary = f"Analysis complete: Average hate ratio = {mean_hate_ratio:.3f} ({len(valid_hate_ratios)}/{total_phrases} phrases analyzed)"
    else:
        summary = f"Analysis complete: {total_phrases} phrases extracted but unable to calculate hate ratios"

    return {
        'overall_classification': overall_classification,
        'phrases': classifications,
        'summary': summary
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
            print(f"     Confidence: {phrase_result['confidence']:.3f}")

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