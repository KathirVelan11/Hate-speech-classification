import pandas as pd
import numpy as np
import pickle
import warnings
import os
import torch
import psutil
import gc
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tqdm import tqdm
import faiss

warnings.filterwarnings('ignore')

def print_memory_usage(label=""):
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
        
print_memory_usage("START")

warnings.filterwarnings('ignore')

print("=" * 80)
print("EXPLAINABLE HATE SPEECH CLASSIFICATION - FAISS K-MEANS CLUSTERING")
print("=" * 80)

# Memory monitoring function
def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

import pandas as pd
import numpy as np
import pickle
import warnings
import os
import torch
import psutil
import gc
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tqdm import tqdm
import faiss

warnings.filterwarnings('ignore')

print("=" * 80)
print("EXPLAINABLE HATE SPEECH CLASSIFICATION - FAISS K-MEANS CLUSTERING")
print("=" * 80)

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

print("\n[1/3] Loading phrase embeddings and data...")

import sys
if os.path.exists('kmeans_clustering.pkl') and os.path.exists('cluster_statistics.csv'):
    print("\nCluster model already exists. Loading from disk.")
    sys.exit(0)

# Check GPU availability for FAISS
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")
if gpu_available:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print(f"Initial Memory Usage: {get_memory_usage():.2f} GB")

# Load phrase embeddings - created as memmap with shape (N, 768)
try:
    # Read memmap with correct shape: 1.77M phrases x 768 dimensions
    phrase_embeddings = np.memmap('phrase_embeddings.npy', dtype='float32', mode='r', shape=(1776478, 768))
    print(f"[OK] Loaded phrase embeddings as memmap: {phrase_embeddings.shape}")
except Exception as e:
    print(f"WARNING: Failed to load as memmap: {e}")
    try:
        # Try complete version
        phrase_embeddings = np.memmap('phrase_embeddings_complete.npy', dtype='float32', mode='r', shape=(1776478, 768))
        print(f"[OK] Loaded COMPLETE phrase embeddings: {phrase_embeddings.shape}")
    except Exception as e2:
        print(f"ERROR: Failed to load embeddings: {e2}")
        print(f"Expected: phrase_embeddings.npy with 1.77M phrases x 768 dimensions")
        raise

# Load unique phrases - COMPLETE VERSION (all 726K dataset)
try:
    with open('unique_phrases_complete.pkl', 'rb') as f:
        unique_phrases = pickle.load(f)
    print(f"[OK] Loaded COMPLETE unique phrases: {len(unique_phrases)} phrases")
except Exception as e:
    print(f"WARNING: Failed to load unique_phrases_complete.pkl: {e}")
    print(f"Falling back to old unique_phrases.pkl...")
    try:
        with open('unique_phrases.pkl', 'rb') as f:
            unique_phrases = pickle.load(f)
        print(f"[OK] Loaded unique phrases: {len(unique_phrases)} phrases")
    except Exception as e2:
        print(f"ERROR: Failed to load any phrases: {e2}")
        raise

# Load phrases dataframe WITH LABELS - COMPLETE VERSION (all 726K dataset)
try:
    phrases_df = pd.read_csv('phrases_with_labels_complete.csv', low_memory=False)
    print(f"[OK] Loaded COMPLETE phrases dataframe WITH LABELS: {len(phrases_df)} rows")
    print(f"  Columns: {list(phrases_df.columns)}")
except Exception as e:
    print(f"WARNING: Could not load phrases_with_labels_complete.csv: {e}")
    print(f"Falling back to old phrases_with_labels.csv...")
    try:
        phrases_df = pd.read_csv('phrases_with_labels.csv', low_memory=False)
        print(f"[OK] Loaded phrases dataframe WITH LABELS: {len(phrases_df)} rows")
        print(f"  Columns: {list(phrases_df.columns)}")
    except Exception as e2:
        print(f"WARNING: Could not load phrases_with_labels.csv: {e2}")
        print(f"Falling back to phrases_with_embeddings.csv (without labels)...")
        try:
            phrases_df = pd.read_csv('phrases_with_embeddings.csv', low_memory=False)
            print(f"[OK] Loaded phrases dataframe: {len(phrases_df)} rows")
        except Exception as e3:
            print(f"ERROR: Failed to load any phrases dataframe: {e3}")
            raise

# VALIDATION: Check if embeddings and phrases match in size
print(f"\n[VALIDATION] Checking data consistency...")
print(f"  Embeddings shape: {phrase_embeddings.shape}")
print(f"  Unique phrases: {len(unique_phrases)}")
print(f"  Phrases dataframe: {len(phrases_df)}")

if phrase_embeddings.shape[0] != len(unique_phrases):
    print(f"\n[WARNING] SIZE MISMATCH DETECTED!")
    print(f"  - Embeddings: {phrase_embeddings.shape[0]} samples")
    print(f"  - Phrases: {len(unique_phrases)} samples")
    
    if phrase_embeddings.shape[0] > 100000:
        print(f"\n[ERROR] Complete embeddings loaded but incomplete phrases!")
        print(f"  Please wait for complete phrase extraction to finish.")
        sys.exit(1)
    else:
        print(f"\n[INFO] Detected OLD incomplete embeddings ({phrase_embeddings.shape[0]} samples)")
        print(f"  Reloading OLD phrases to match...")
        
        # Reload OLD unique phrases (not complete)
        try:
            with open('unique_phrases.pkl', 'rb') as f:
                unique_phrases = pickle.load(f)
            print(f"[OK] Reloaded OLD unique phrases: {len(unique_phrases)} phrases")
            
            # Also reload OLD phrases dataframe
            try:
                phrases_df = pd.read_csv('phrases_with_labels.csv', low_memory=False)
                print(f"[OK] Reloaded OLD phrases dataframe: {len(phrases_df)} rows")
            except:
                pass
                
        except Exception as e:
            print(f"[WARNING] Could not reload old unique_phrases: {e}")
        
        if phrase_embeddings.shape[0] == len(unique_phrases):
            print(f"[OK] Data is now consistent at {len(unique_phrases)} samples")
        else:
            print(f"\n[ERROR] Still mismatched after reload!")
            print(f"  - Embeddings: {phrase_embeddings.shape[0]} samples")
            print(f"  - Phrases: {len(unique_phrases)} samples")
            sys.exit(1)
else:
    print(f"[OK] Data is consistent! Using {len(unique_phrases)} phrases with embeddings.")

print("\n[2/3] Training FAISS K-Means clustering with GPU support...")

try:
    print(f"Memory before normalization: {get_memory_usage():.2f} GB")
    
    print("\n📊 Normalizing embeddings...")
    norm_start = time.time()
    batch_size = 10000
    normalized_embeddings = np.zeros_like(phrase_embeddings, dtype=np.float32)
    
    with tqdm(total=len(phrase_embeddings), desc="Normalizing", unit="samples") as pbar:
        for i in range(0, len(phrase_embeddings), batch_size):
            end_idx = min(i + batch_size, len(phrase_embeddings))
            normalized_embeddings[i:end_idx] = normalize(phrase_embeddings[i:end_idx], norm='l2').astype('float32')
            pbar.update(end_idx - i)
    
    norm_time = time.time() - norm_start
    print(f"✓ Normalization completed in {norm_time:.2f} seconds")
    print(f"  Memory after normalization: {get_memory_usage():.2f} GB")
    
    # Clean up original embeddings to save memory
    del phrase_embeddings
    gc.collect()
    
    # Elbow method for optimal k with progress tracking
    print("\n📈 Performing hyperparameter tuning using Elbow Method...")
    print("  Using subset of data for faster computation...")
    
    subset_size = min(int(0.1 * len(normalized_embeddings)), 15000)
    np.random.seed(42)
    subset_indices = np.random.choice(len(normalized_embeddings), subset_size, replace=False)
    subset_embeddings = normalized_embeddings[subset_indices]
    
    print(f"  Subset size: {subset_size:,} samples ({subset_size/len(normalized_embeddings)*1000:.1f}%)")
    print(f"  Memory usage: {get_memory_usage():.2f} GB")
    
    inertia = []
    k_range = range(2, 16)
    
    print("\n🔍 Testing different k values...")
    with tqdm(total=len(k_range), desc="Elbow method", unit="k") as pbar:
        for k in k_range:
            kmeans = faiss.Kmeans(
                d=subset_embeddings.shape[1],
                k=k,
                niter=10,
                verbose=False,
                gpu=gpu_available,
                seed=42
            )
            kmeans.train(subset_embeddings)
            inertia.append(kmeans.obj[-1])
            pbar.set_postfix({'k': k, 'inertia': f'{kmeans.obj[-1]:.2e}'})
            pbar.update(1)
    
    print(f"✓ Elbow method completed")
    gc.collect()
    
    # Plot elbow curve
    print("\n📊 Generating elbow curve...")
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'elbow_method.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved elbow curve to {plots_dir}/elbow_method.png")
    
    # Choose optimal k (can be adjusted based on elbow plot)
    optimal_k = 10
    print(f"\n🎯 Selected optimal k: {optimal_k}")
    
    # Clean up subset
    del subset_embeddings
    gc.collect()
    
    # FAISS K-Means clustering on full dataset
    print(f"\n🚀 Training FAISS K-Means with k={optimal_k}...")
    print(f"  Dataset size: {len(normalized_embeddings):,} samples")
    print(f"  Embedding dimension: {normalized_embeddings.shape[1]}")
    print(f"  Using GPU: {gpu_available}")
    print(f"  Memory before clustering: {get_memory_usage():.2f} GB")
    
    cluster_start = time.time()
    
    print("\n⏳ Training K-Means (using CPU, this may take 2-5 minutes)...")
    
    torch.cuda.empty_cache() if gpu_available else None
    
    # Initialize FAISS K-Means (using CPU - faster and more stable than GPU for this size)
    clusterer = faiss.Kmeans(
        d=normalized_embeddings.shape[1],
        k=optimal_k,
        niter=20,
        verbose=True,
        gpu=False,  # Use CPU for stability
        seed=42
    )
    
    print("=" * 60)
    
    # Train with progress updates
    clusterer.train(normalized_embeddings)
    
    print("=" * 60)
    print(f"✓ Training completed!")
    
    # Get cluster assignments with progress bar
    print("\n🏷️  Assigning samples to clusters...")
    _, cluster_labels = clusterer.assign(normalized_embeddings)
    
    cluster_time = time.time() - cluster_start
    
    print(f"\n✓ Clustering completed in {cluster_time:.2f} seconds")
    print(f"  Processing speed: {cluster_time/len(normalized_embeddings)*100:.2f} ms per sample")
    print(f"  Memory after clustering: {get_memory_usage():.2f} GB")
    
    del normalized_embeddings
    gc.collect()
    
    # Add cluster labels
    print("\n📋 Creating cluster assignments dataframe...")
    phrase_clusters = pd.DataFrame({
        'phrase': unique_phrases,
        'cluster': cluster_labels
    })
    
    n_clusters = len(set(cluster_labels))
    n_noise = 0  # K-Means has no noise points
    
    print(f"\n✓ Clustering complete!")
    print(f"  - Number of clusters: {n_clusters}")
    print(f"  - Noise points: {n_noise}")
    print(f"  - Average cluster size: {len(cluster_labels) / n_clusters:.1f}")
    print(f"  - Cluster labels (first 20): {cluster_labels[:20]}")
    
    phrases_df = phrases_df.merge(phrase_clusters, on='phrase', how='left')
    
    # Final statistics
    print(f"\n📊 Final Memory Usage: {get_memory_usage():.2f} GB")
    if gpu_available:
        print(f"📊 GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
except Exception as e:
    print("ERROR:", f"Failed during clustering: {e}")
    import traceback
    traceback.print_exc()
    raise


print("\n" + "=" * 80)
print("[3/3] Computing cluster statistics and saving model...")
print("=" * 80)

cluster_stats = []

# Group by cluster and compute statistics
grouped = phrases_df.groupby('cluster')

for cluster_id, cluster_data in grouped:
    total_phrases = len(cluster_data)
    
    # Basic frequency stats
    total_count = cluster_data['count'].sum() if 'count' in cluster_data.columns else 0
    avg_count = cluster_data['count'].mean() if 'count' in cluster_data.columns else 0
    sample_phrases = cluster_data.nlargest(5, 'count')['phrase'].tolist() if 'count' in cluster_data.columns else cluster_data['phrase'].head(5).tolist()
    
    # Label-based stats (if available)
    if 'hate_count' in cluster_data.columns and 'non_hate_count' in cluster_data.columns:
        total_hate_count = cluster_data['hate_count'].sum()
        total_non_hate_count = cluster_data['non_hate_count'].sum()
        total_labeled = total_hate_count + total_non_hate_count
        cluster_hate_ratio = total_hate_count / total_labeled if total_labeled > 0 else 0
        cluster_hate_indicator = 'HIGH HATE' if cluster_hate_ratio > 0.66 else ('LOW HATE' if cluster_hate_ratio < 0.33 else 'NEUTRAL')
    else:
        total_hate_count = None
        total_non_hate_count = None
        cluster_hate_ratio = None
        cluster_hate_indicator = 'N/A'
    
    stats_dict = {
        'cluster_id': cluster_id,
        'size': total_phrases,
        'total_phrase_frequency': total_count,
        'avg_phrase_frequency': avg_count,
        'sample_phrases': ', '.join(sample_phrases[:3])  # Top 3 samples
    }
    
    # Add label info if available
    if total_hate_count is not None:
        stats_dict.update({
            'hate_count': total_hate_count,
            'non_hate_count': total_non_hate_count,
            'cluster_hate_ratio': f"{cluster_hate_ratio:.1%}",
            'hate_indicator': cluster_hate_indicator
        })
    
    cluster_stats.append(stats_dict)

cluster_stats_df = pd.DataFrame(cluster_stats)

print("\nCluster Statistics:")
print("=" * 80)

# Display basic stats
basic_cols = ['cluster_id', 'size', 'total_phrase_frequency', 'avg_phrase_frequency']
print(f"\n{cluster_stats_df[basic_cols].to_string()}")

# Display label stats if available
if 'hate_indicator' in cluster_stats_df.columns:
    print("\n\nCluster Hate Speech Analysis (from extracted labels):")
    print("=" * 80)
    label_cols = ['cluster_id', 'hate_count', 'non_hate_count', 'cluster_hate_ratio', 'hate_indicator']
    print(f"\n{cluster_stats_df[label_cols].to_string()}")

try:
    phrases_df.to_csv('phrases_with_clusters.csv', index=False)
    print("\n✓ Saved phrases_with_clusters.csv")
except Exception as e:
    print("ERROR:", f"Failed to save phrases CSV: {e}")
    raise

try:
    cluster_stats_df.to_csv('cluster_statistics.csv', index=False)
    print("✓ Saved cluster_statistics.csv")
except Exception as e:
    print("ERROR:", f"Failed to save cluster statistics: {e}")
    raise

try:
    kmeans_model = {
        'centroids': clusterer.centroids,
        'k': optimal_k,
        'obj': clusterer.obj,
        'unique_phrases': unique_phrases
    }
    with open('kmeans_clustering.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    print("✓ Saved kmeans_clustering.pkl")
except Exception as e:
    print("ERROR:", f"Failed to save clustering model: {e}")
    raise

print("\n" + "=" * 80)
print("CLUSTERING COMPLETE!")
print("=" * 80)
print("\nFinal Statistics:")
print(f"  - Total clusters: {n_clusters}")
print(f"  - Noise points: {n_noise}")
print(f"  - Total phrases: {len(phrases_df)}")
print(f"  - Unique phrases: {phrases_df['phrase'].nunique()}")

print("\n📁 Saved Model Files:")
print("  ✓ kmeans_clustering.pkl")
print("  ✓ phrases_with_clusters.csv")
print("  ✓ cluster_statistics.csv")

print("\n📊 Next Steps:")
print("  Run plotting.py to visualize clustering results")
print("  Run inference.py to classify new text samples")

print("\n" + "=" * 80)
print_memory_usage("END")
