
import pandas as pd
import numpy as np
import pickle
import warnings
import os
import csv
import torch
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

# Hugging Face datasets for efficient loading
from datasets import load_from_disk, Dataset

# KeyBERT for phrase extraction
from keybert import KeyBERT

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

# Utilities
from collections import Counter
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - REDUCED FOR STABILITY
# ============================================================================

# Extraction settings
TOP_K = 3  # Reduced from 5
MIN_NGRAM = 1
MAX_NGRAM = 4
DIVERSITY = 0.5

# Batch settings - MUCH SMALLER
EXTRACTION_BATCH_SIZE = 100  # Reduced from 2000
EMBEDDING_BATCH_SIZE = 32 if torch.cuda.is_available() else 16  # Reduced from 1024/256
ENCODE_BATCH_SIZE = 16 if torch.cuda.is_available() else 8  # Reduced from 256/64

# Parallel processing - MINIMAL
NUM_WORKERS = 2  # Reduced from cpu_count()-2

# ADD THIS - Memory management
import gc
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# STEP 3: LOAD MODELS AND DATASET
# ============================================================================

print("=" * 80)
print("EXPLAINABLE HATE SPEECH CLASSIFICATION - PHRASE EXTRACTION (OPTIMIZED)")
print("=" * 80)

print("\n[1/4] Loading models and custom stop words...")

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Load custom stop words
try:
    with open('custom_stop_words.pkl', 'rb') as f:
        custom_stop_words = pickle.load(f)
    print(f"Loaded custom stop words: {len(custom_stop_words)} words")
except Exception as e:
    print(f"ERROR: Failed to load custom stop words: {e}")
    raise

# Load model info
try:
    with open('embedding_model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    print(f"\nModel info loaded:")
    print(f"  - Model path: {model_info['model_path']}")
    print(f"  - Fine-tuned: {model_info['fine_tuned']}")
    print(f"  - LoRA applied: {model_info['lora_applied']}")
except Exception as e:
    print(f"ERROR: Failed to load model info: {e}")
    raise

# Initialize KeyBERT with GPU support
try:
    kw_model = KeyBERT(model='paraphrase-mpnet-base-v2')
    print(f"\nKeyBERT model loaded on {'GPU' if device == 'cuda' else 'CPU'}")
except Exception as e:
    print(f"ERROR: Failed to load KeyBERT model: {e}")
    raise

# Load fine-tuned embedding model
try:
    if model_info['fine_tuned']:
        embedding_model = SentenceTransformer(model_info['model_path'], device=device)
        print(f"Fine-tuned embedding model loaded from {model_info['model_path']}")
    else:
        embedding_model = SentenceTransformer(model_info['base_model'], device=device)
        print(f"Base embedding model loaded: {model_info['base_model']}")
except Exception as e:
    print(f"ERROR: Failed to load embedding model: {e}")
    raise

# Load dataset
print("\n[2/4] Loading dataset...")
try:
    dataset = load_from_disk('full_dataset')
    print(f"Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"ERROR: Failed to load dataset: {e}")
    raise


# ============================================================================
# STEP 4: OPTIMIZED PHRASE EXTRACTION
# ============================================================================

print("\n[3/4] Defining optimized phrase extraction functions...")

# Hate-relevant words for filtering (expanded)
HATE_RELEVANT_WORDS = {
    'hate', 'love', 'like', 'dislike', 'fear', 'racist', 'sexist',
    'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', "n't",
    'very', 'really', 'so', 'too', 'quite', 'extremely', 'totally', 'completely',
    'should', 'must', 'ought', 'need', 'have', 'has', 'had',
    'all', 'every', 'each', 'any', 'some', 'few', 'many', 'most', 'several',
    'they', 'them', 'their', 'those', 'these', 'we', 'us', 'our',
    'back', 'away', 'out', 'off', 'down',
    'more', 'less', 'better', 'worse', 'best', 'worst',
    'only', 'just', 'still', 'even', 'again', 'against',
    'fuck', 'shit', 'damn', 'hell', 'stupid', 'idiot', 'retard',
    'kill', 'die', 'death', 'dead', 'attack', 'violence', 'violent'
}


def extract_key_phrases_fast(text, label, top_k=TOP_K):
    """
    OPTIMIZED: Extract key phrases using KeyBERT only.
    Removed spaCy dependency for 90% speedup.
    """
    # Adjust max_ngram based on sentence length
    sentence_length = len(text.split())
    if sentence_length > 20:
        max_ngram = 4
    elif sentence_length > 15:
        max_ngram = 3
    else:
        max_ngram = 2
    
    # Extract candidates using KeyBERT
    try:
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(MIN_NGRAM, max_ngram),
            stop_words=list(custom_stop_words),
            top_n=top_k * 2,  # Extract more for filtering
            diversity=DIVERSITY
        )
    except:
        return []
    
    # Simple filtering: prioritize phrases with hate-relevant words
    filtered_phrases = []
    for phrase, score in keywords:
        phrase_lower = phrase.lower().strip()
        phrase_words = set(phrase_lower.split())
        
        # Boost score if contains hate-relevant words
        has_relevant = bool(phrase_words & HATE_RELEVANT_WORDS)
        adjusted_score = score * 1.2 if has_relevant else score
        
        filtered_phrases.append((phrase_lower, adjusted_score))
    
    # Sort by adjusted score and take top-k
    filtered_phrases = sorted(filtered_phrases, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Return phrase information
    results = []
    for phrase, score in filtered_phrases:
        results.append({
            'phrase': phrase,
            'score': score,
            'label': label,
            'original_text': text
        })
    
    return results


def process_batch_worker(batch_data):
    """Worker function with memory management."""
    all_phrases = []
    for text, label in batch_data:
        try:
            phrases = extract_key_phrases_fast(text, label)
            all_phrases.extend(phrases)
        except Exception as e:
            continue
    
    # Clear memory after each batch
    gc.collect()
    return all_phrases


print("Optimized phrase extraction functions defined!")


# ============================================================================
# STEP 5: EXTRACT PHRASES WITHOUT MULTIPROCESSING
# ============================================================================

print(f"\n[4/4] Extracting key phrases WITHOUT multiprocessing...")
print(f"Batch size: {EXTRACTION_BATCH_SIZE}")

phrases_output_file = 'extracted_phrases.csv'

# Initialize CSV file
try:
    with open(phrases_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['phrase', 'score', 'label', 'original_text'])
        writer.writeheader()
    print(f"Initialized output file: {phrases_output_file}")
except Exception as e:
    print(f"ERROR: Failed to initialize output file: {e}")
    raise

# Prepare data for sequential processing
print("Preparing batches for sequential processing...")
total_samples = len(dataset)
all_batches = []

for i in range(0, total_samples, EXTRACTION_BATCH_SIZE):
    batch = dataset[i:i + EXTRACTION_BATCH_SIZE]
    batch_data = list(zip(batch['text'], batch['label']))
    all_batches.append(batch_data)

print(f"Created {len(all_batches)} batches")

# Process batches sequentially
total_phrases_count = 0
label_counts = Counter()

print("\nProcessing batches sequentially...")
results = []
for batch_data in tqdm(all_batches, desc="Extracting phrases"):
    batch_phrases = process_batch_worker(batch_data)
    results.append(batch_phrases)

    # Write results to CSV
    if batch_phrases:
        try:
            with open(phrases_output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['phrase', 'score', 'label', 'original_text'])
                writer.writerows(batch_phrases)
            
            total_phrases_count += len(batch_phrases)
            for phrase_data in batch_phrases:
                label_counts[phrase_data['label']] += 1
        except Exception as e:
            print(f"Warning: Failed to write batch: {e}")
            continue

print(f"\nExtraction complete!")
print(f"  - Total phrases extracted: {total_phrases_count}")
print(f"  - Saved to: {phrases_output_file}")
print(f"\nPhrase distribution by label:")
for label, count in sorted(label_counts.items()):
    print(f"  - Label {label}: {count}")

# Load phrases for embedding generation
print("\nLoading extracted phrases...")
try:
    phrases_df = pd.read_csv(phrases_output_file, low_memory=False)
    print(f"  - Total phrases: {len(phrases_df)}")
    print(f"  - Unique phrases: {phrases_df['phrase'].nunique()}")
except Exception as e:
    print(f"ERROR: Failed to load extracted phrases: {e}")
    raise


# ============================================================================
# STEP 6: GENERATE PHRASE EMBEDDINGS (GPU-OPTIMIZED)
# ============================================================================

print(f"\n[5/4] Computing phrase embeddings using fine-tuned model...")

unique_phrases = phrases_df['phrase'].unique()
print(f"Computing embeddings for {len(unique_phrases)} unique phrases...")
print(f"Using device: {device}")
print(f"Embedding batch size: {EMBEDDING_BATCH_SIZE}")

# Generate embeddings in large batches with GPU acceleration
phrase_embeddings_list = []

try:
    num_batches = (len(unique_phrases) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    
    with tqdm(total=num_batches, desc="Computing embeddings") as pbar:
        for i in range(0, len(unique_phrases), EMBEDDING_BATCH_SIZE):
            batch = unique_phrases[i:i+EMBEDDING_BATCH_SIZE].tolist()
            
            try:
                batch_embeddings = embedding_model.encode(
                    batch,
                    show_progress_bar=False,
                    batch_size=ENCODE_BATCH_SIZE,
                    convert_to_numpy=True,
                    device=device,
                    normalize_embeddings=True
                )
                phrase_embeddings_list.append(batch_embeddings)
                pbar.update(1)
                
                # ADD THIS - Clear GPU cache every 10 batches
                if i % (EMBEDDING_BATCH_SIZE * 10) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
            except Exception as e:
                print(f"\nWarning: Batch encoding failed at index {i}: {e}")
                if device == 'cuda':
                    print("Retrying with CPU...")
                    try:
                        batch_embeddings = embedding_model.encode(
                            batch,
                            show_progress_bar=False,
                            batch_size=32,
                            convert_to_numpy=True,
                            device='cpu',
                            normalize_embeddings=True
                        )
                        phrase_embeddings_list.append(batch_embeddings)
                        pbar.update(1)
                    except Exception as e2:
                        print(f"CPU encoding also failed: {e2}")
                        raise
                else:
                    raise
    
    # Concatenate all embeddings
    phrase_embeddings = np.vstack(phrase_embeddings_list)
    del phrase_embeddings_list  # Free memory
    
    print(f"\nEmbeddings computed!")
    print(f"  - Embedding shape: {phrase_embeddings.shape}")
    print(f"  - Embedding dimension: {phrase_embeddings.shape[1]}")
    print(f"  - Memory usage: ~{phrase_embeddings.nbytes / 1024**2:.2f} MB")
    
except Exception as e:
    print(f"ERROR: Failed to compute embeddings: {e}")
    raise

# Create mapping
phrases_df['embedding_idx'] = phrases_df['phrase'].map(
    {phrase: idx for idx, phrase in enumerate(unique_phrases)}
)

# Save embeddings and phrase mappings
print("\nSaving embeddings and mappings...")
try:
    np.save('phrase_embeddings.npy', phrase_embeddings)
    print("  - Saved embeddings: phrase_embeddings.npy")
    
    with open('unique_phrases.pkl', 'wb') as f:
        pickle.dump(unique_phrases, f)
    print("  - Saved unique phrases: unique_phrases.pkl")
    
    phrases_df.to_csv('phrases_with_embeddings.csv', index=False)
    print("  - Saved phrases dataframe: phrases_with_embeddings.csv")
    
except Exception as e:
    print(f"ERROR: Failed to save embeddings: {e}")
    raise


print("\n" + "=" * 80)
print("PHRASE EXTRACTION AND EMBEDDING COMPLETE!")
print("=" * 80)
print("\nSaved files:")
print("  - Extracted phrases: extracted_phrases.csv")
print("  - Phrase embeddings: phrase_embeddings.npy")
print("  - Unique phrases: unique_phrases.pkl")
print("  - Phrases with embeddings: phrases_with_embeddings.csv")
print("\nNext step: Run clustering_and_plotting.py")