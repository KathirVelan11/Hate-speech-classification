import pandas as pd
import numpy as np
import pickle
import warnings
import os
import torch
import psutil
import shutil
import gc
from multiprocessing import cpu_count
warnings.filterwarnings('ignore')

from collections import Counter, defaultdict
from tqdm import tqdm

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ============================================================================
# MEMORY MONITORING
# ============================================================================

def print_memory_usage(label: str = "") -> None:
    """Print current RAM and GPU memory usage."""
    prefix = f"[{label}] " if label else ""
    process = psutil.Process()
    ram_used = process.memory_info().rss / 1024 ** 3
    ram_avail = psutil.virtual_memory().available / 1024 ** 3
    
    print(f"\n{prefix}[MEMORY] Usage:")
    print(f"  RAM Used  : {ram_used:.2f} GB")
    print(f"  RAM Avail : {ram_avail:.2f} GB")

    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / 1024 ** 3
        gpu_reserv = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"  GPU Alloc : {gpu_alloc:.2f} GB")
        print(f"  GPU Rsvrd : {gpu_reserv:.2f} GB")


def check_disk_space(path: str, required_gb: float) -> None:
    """Raise a RuntimeError if insufficient disk space."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / 1024 ** 3
    if free_gb < required_gb:
        raise RuntimeError(f"Insufficient disk space at '{path}': {free_gb:.2f} GB available, {required_gb} GB required.")


print_memory_usage("START")
check_disk_space('.', 50)

# ============================================================================
# CONFIGURATION
# ============================================================================

TOP_K = 3
MIN_NGRAM = 1
MAX_NGRAM = 4
DIVERSITY = 0.3

EXTRACTION_BATCH_SIZE = 1000
EMBEDDING_BATCH_SIZE = 128
ENCODE_BATCH_SIZE = 64
FREQUENCY_THRESHOLD = 3

CHECKPOINT_INTERVAL = 50000  # Save checkpoint every 50K samples

torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("=" * 80)
print("EXPLAINABLE HATE SPEECH CLASSIFICATION - COMPLETE PHRASE EXTRACTION PIPELINE")
print("=" * 80)

# ============================================================================
# CHECK IF FINAL FILES ALREADY EXIST
# ============================================================================

if os.path.exists('unique_phrases.pkl') and os.path.exists('phrase_embeddings.npy') and os.path.exists('phrases_with_labels.csv'):
    print("\n[COMPLETE] All final files already exist. Loading from disk.")
    with open('unique_phrases.pkl', 'rb') as f:
        unique_phrases = pickle.load(f)
    phrase_embeddings = np.load('phrase_embeddings.npy', mmap_mode='r')
    labels_df = pd.read_csv('phrases_with_labels.csv')
    print(f"✓ Loaded {len(unique_phrases)} unique phrases with embeddings and labels!")
    print_memory_usage("END")
    exit(0)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

print("\n[1/5] Loading models and custom stop words...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load custom stop words
try:
    with open('custom_stop_words.pkl', 'rb') as f:
        custom_stop_words = pickle.load(f)
    print(f"✓ Loaded {len(custom_stop_words)} custom stop words")
except Exception as e:
    print(f"ERROR: Failed to load custom stop words: {e}")
    custom_stop_words = set()

# Load embedding model
try:
    # Try fine-tuned model first
    if os.path.exists('embedding_model_info.pkl'):
        with open('embedding_model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        if model_info.get('fine_tuned', False) and os.path.exists(model_info['model_path']):
            embedding_model = SentenceTransformer(model_info['model_path'], device=device)
            print(f"✓ Loaded fine-tuned embedding model from {model_info['model_path']}")
        else:
            embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)
            print(f"✓ Loaded base embedding model: paraphrase-mpnet-base-v2")
    else:
        embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)
        print(f"✓ Loaded base embedding model: paraphrase-mpnet-base-v2")
except Exception as e:
    print(f"ERROR: Failed to load embedding model: {e}")
    raise

# Initialize KeyBERT
try:
    kw_model = KeyBERT(model=embedding_model)
    print(f"✓ KeyBERT model initialized on {'GPU' if device == 'cuda' else 'CPU'}")
except Exception as e:
    print(f"ERROR: Failed to initialize KeyBERT: {e}")
    raise

# ============================================================================
# HATE-RELEVANT WORDS FOR PHRASE FILTERING
# ============================================================================

HATE_RELEVANT_WORDS = {
    'hate', 'love', 'like', 'dislike', 'fear', 'racist', 'sexist',
    'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', "n't",
    'very', 'really', 'so', 'too', 'quite', 'extremely', 'totally', 'completely',
    'should', 'must', 'ought', 'need', 'have', 'has', 'had',
    'all', 'every', 'each', 'any', 'some', 'few', 'many', 'most', 'several',
    'they', 'them', 'their', 'those', 'these', 'we', 'us', 'our',
    'back', 'away', 'out', 'off', 'down',
    'more', 'less', 'better', 'worse', 'best', 'worst',
    'only', 'just', 'still', 'even', 'again', 'against'
}

def extract_key_phrases_fast(text, top_k=TOP_K):
    """Extract key phrases with hate-relevant word boosting."""
    if not text or len(text.strip()) < 3:
        return []
    
    sentence_length = len(text.split())
    if sentence_length > 20:
        max_ngram = 4
    elif sentence_length > 15:
        max_ngram = 3
    else:
        max_ngram = 2
    
    try:
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(MIN_NGRAM, max_ngram),
            stop_words=list(custom_stop_words),
            top_n=top_k * 2,
            diversity=DIVERSITY
        )
    except:
        return []
    
    filtered_phrases = []
    for phrase, score in keywords:
        phrase_lower = phrase.lower().strip()
        phrase_words = set(phrase_lower.split())
        has_relevant = bool(phrase_words & HATE_RELEVANT_WORDS)
        adjusted_score = score * 1.2 if has_relevant else score
        filtered_phrases.append((phrase_lower, adjusted_score))
    
    filtered_phrases = sorted(filtered_phrases, key=lambda x: x[1], reverse=True)[:top_k]
    return [p[0] for p in filtered_phrases]

# ============================================================================
# LOAD DATASET
# ============================================================================

print("\n[2/5] Loading dataset...")
DATASET_PATH = 'HateSpeechDatasetBalanced.csv'

try:
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"✓ Dataset loaded: {len(df)} samples")
except Exception as e:
    print(f"ERROR: Failed to read dataset from {DATASET_PATH}: {e}")
    raise

# ============================================================================
# PASS 1: PHRASE EXTRACTION AND FREQUENCY COUNTING
# ============================================================================

print("\n[3/5] PASS 1: Phrase extraction and frequency counting...")

CHECKPOINT_FILE = 'extraction_checkpoint.pkl'
phrase_dict = {}
start_idx = 0

# Load checkpoint if exists
if os.path.exists(CHECKPOINT_FILE):
    print("Loading checkpoint...")
    with open(CHECKPOINT_FILE, 'rb') as f:
        checkpoint = pickle.load(f)
        phrase_dict = checkpoint.get('phrase_dict', {})
        start_idx = checkpoint.get('sample_count', 0)
    print(f"✓ Resumed from sample {start_idx} with {len(phrase_dict)} phrases")

print(f"Processing samples {start_idx} to {len(df)}...")
start_time = __import__('time').time()

try:
    for idx in range(start_idx, len(df)):
        text = str(df.iloc[idx]['text'])
        label = int(df.iloc[idx]['label'])
        
        phrases = extract_key_phrases_fast(text)
        for p in phrases:
            if p not in phrase_dict:
                phrase_dict[p] = {
                    'count': 0,
                    'hate_count': 0,
                    'non_hate_count': 0
                }
            phrase_dict[p]['count'] += 1
            if label == 1:
                phrase_dict[p]['hate_count'] += 1
            else:
                phrase_dict[p]['non_hate_count'] += 1
        
        # Progress tracking
        current = idx + 1
        if current % 10000 == 0:
            elapsed = __import__('time').time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining = (len(df) - current) / rate if rate > 0 else 0
            print(f"  Processed {current}/{len(df)} samples ({current/len(df)*100:.1f}%) "
                  f"- Unique phrases: {len(phrase_dict)} - ETA: {remaining/3600:.1f}h")
        
        # Save checkpoint
        if current % CHECKPOINT_INTERVAL == 0 and current > start_idx:
            checkpoint_data = {
                'sample_count': current,
                'phrase_dict': phrase_dict
            }
            with open(CHECKPOINT_FILE, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"  [CHECKPOINT] Saved progress at {current} samples")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Saving checkpoint before exit...")
    checkpoint_data = {
        'sample_count': idx,
        'phrase_dict': phrase_dict
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved at sample {idx}. Resume later.")
    exit(0)
except Exception as e:
    print(f"ERROR during extraction: {e}")
    # Save emergency checkpoint
    checkpoint_data = {
        'sample_count': idx if 'idx' in locals() else start_idx,
        'phrase_dict': phrase_dict
    }
    with open('extraction_checkpoint_emergency.pkl', 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Emergency checkpoint saved.")
    raise

# Save final checkpoint
final_checkpoint = {
    'sample_count': len(df),
    'phrase_dict': phrase_dict
}
with open(CHECKPOINT_FILE, 'wb') as f:
    pickle.dump(final_checkpoint, f)
print(f"✓ Final checkpoint saved with {len(phrase_dict)} phrases")

# ============================================================================
# FILTER PHRASES BY FREQUENCY
# ============================================================================

print(f"\n[3.5/5] Filtering phrases (minimum frequency: {FREQUENCY_THRESHOLD})...")

frequent_phrases = {
    p: data for p, data in phrase_dict.items() 
    if data['count'] >= FREQUENCY_THRESHOLD
}

unique_phrases = list(frequent_phrases.keys())
print(f"  Total unique phrases found: {len(phrase_dict)}")
print(f"  Phrases with count >= {FREQUENCY_THRESHOLD}: {len(unique_phrases)}")

# Save phrase frequencies
freq_data = []
for phrase, data in frequent_phrases.items():
    total = data['count']
    hate = data['hate_count']
    non_hate = data['non_hate_count']
    hate_ratio = hate / total if total > 0 else 0
    
    if hate_ratio > 0.66:
        indicator = 'HIGH HATE'
    elif hate_ratio > 0.33:
        indicator = 'NEUTRAL'
    else:
        indicator = 'LOW HATE'
    
    freq_data.append({
        'phrase': phrase,
        'total_frequency': total,
        'hate_count': hate,
        'non_hate_count': non_hate,
        'hate_ratio': f"{hate_ratio*100:.1f}%",
        'hate_indicator': indicator
    })

freq_df = pd.DataFrame(freq_data)
freq_df = freq_df.sort_values('total_frequency', ascending=False)
freq_df.to_csv('extracted_phrases.parquet' if False else 'extracted_phrases.csv', index=False)
print("✓ Saved phrase frequencies to extracted_phrases.csv")

# ============================================================================
# PASS 2: EMBEDDING GENERATION
# ============================================================================

print(f"\n[4/5] PASS 2: Generating embeddings for {len(unique_phrases)} phrases...")

embedding_dim = 768
phrase_embeddings = np.memmap(
    'phrase_embeddings.npy', 
    dtype='float32', 
    mode='w+', 
    shape=(len(unique_phrases), embedding_dim)
)

try:
    chunk_size = 10000
    total_phrases = len(unique_phrases)
    
    with tqdm(total=total_phrases, desc="Embedding phrases", unit="phrases") as pbar:
        for i in range(0, total_phrases, chunk_size):
            chunk_end = min(i + chunk_size, total_phrases)
            chunk = unique_phrases[i:chunk_end]
            
            # Encode with error handling
            try:
                chunk_embeddings = embedding_model.encode(
                    chunk,
                    batch_size=ENCODE_BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            except Exception as e:
                # Retry with smaller batch size
                print(f"\n  Retrying chunk {i}-{chunk_end} with batch_size=16...")
                chunk_embeddings = embedding_model.encode(
                    chunk,
                    batch_size=16,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            
            # Normalize and save
            chunk_embeddings = normalize(chunk_embeddings, norm='l2', axis=1)
            phrase_embeddings[i:chunk_end] = chunk_embeddings
            pbar.update(len(chunk))
            
            # Memory cleanup
            if i % (chunk_size * 5) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    phrase_embeddings.flush()
    print(f"✓ All {total_phrases} embeddings saved to phrase_embeddings.npy")

except Exception as e:
    print(f"ERROR generating embeddings: {e}")
    raise

# ============================================================================
# SAVE FINAL FILES
# ============================================================================

print(f"\n[5/5] Saving final files...")

# Save unique phrases
with open('unique_phrases.pkl', 'wb') as f:
    pickle.dump(unique_phrases, f)
print("✓ Saved unique_phrases.pkl")

# Save phrases with labels (complete)
freq_df.to_csv('phrases_with_labels.csv', index=False)
print("✓ Saved phrases_with_labels.csv")

# Save phrases with embeddings index
phrases_index_df = pd.DataFrame({
    'phrase': unique_phrases,
    'index': range(len(unique_phrases))
})
phrases_index_df.to_csv('phrases_with_embeddings.csv', index=False)
print("✓ Saved phrases_with_embeddings.csv")

# Clean up checkpoint file after successful completion
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("✓ Cleaned up checkpoint file")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PHRASE EXTRACTION PIPELINE COMPLETE!")
print("=" * 80)
print(f"\nFinal Statistics:")
print(f"  - Total samples processed: {len(df)}")
print(f"  - Unique phrases extracted: {len(unique_phrases)}")
print(f"  - Embedding dimension: {embedding_dim}")

hate_count = len(freq_df[freq_df['hate_indicator'] == 'HIGH HATE'])
neutral_count = len(freq_df[freq_df['hate_indicator'] == 'NEUTRAL'])
non_hate_count = len(freq_df[freq_df['hate_indicator'] == 'LOW HATE'])

print(f"  - HIGH HATE phrases: {hate_count}")
print(f"  - NEUTRAL phrases: {neutral_count}")
print(f"  - LOW HATE phrases: {non_hate_count}")

print(f"\nGenerated Files:")
print(f"  - extracted_phrases.csv (phrase frequencies and labels)")
print(f"  - unique_phrases.pkl (list of unique phrases)")
print(f"  - phrase_embeddings.npy (embedding vectors)")
print(f"  - phrases_with_embeddings.csv (phrase index mapping)")
print(f"  - phrases_with_labels.csv (complete label statistics)")

print(f"\nNext Steps:")
print(f"  1. Run clustering.py to cluster the phrases")
print(f"  2. Run inference.py for hate speech classification")

print_memory_usage("END")
print(f"\nCompleted at: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")