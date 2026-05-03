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

# Check GPU availability for FAISS
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")
if gpu_available:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print(f"Initial Memory Usage: {get_memory_usage():.2f} GB")

# Load phrase embeddings
try:
    phrase_embeddings = np.load('phrase_embeddings.npy')
    print(f"Loaded phrase embeddings: {phrase_embeddings.shape}")
except Exception as e:
    print(f"ERROR: Failed to load phrase embeddings: {e}")
    raise

# Load unique phrases
try:
    with open('unique_phrases.pkl', 'rb') as f:
        unique_phrases = pickle.load(f)
    print(f"Loaded unique phrases: {len(unique_phrases)} phrases")
except Exception as e:
    print(f"ERROR: Failed to load unique phrases: {e}")
    raise

# Load phrases dataframe
try:
    phrases_df = pd.read_csv('phrases_with_embeddings.csv', low_memory=False)
    print(f"Loaded phrases dataframe: {len(phrases_df)} rows")
except Exception as e:
    print(f"ERROR: Failed to load phrases dataframe: {e}")
    raise

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
    
    subset_size = min(int(0.1 * len(normalized_embeddings)), 50000)
    np.random.seed(42)
    subset_indices = np.random.choice(len(normalized_embeddings), subset_size, replace=False)
    subset_embeddings = normalized_embeddings[subset_indices]
    
    print(f"  Subset size: {subset_size:,} samples ({subset_size/len(normalized_embeddings)*1000:.1f}%)")
    print(f"  Memory usage: {get_memory_usage():.2f} GB")
    
    inertia = []
    k_range = range(2, 21)
    
    print("\n🔍 Testing different k values...")
    with tqdm(total=len(k_range), desc="Elbow method", unit="k") as pbar:
        for k in k_range:
            kmeans = faiss.Kmeans(
                d=subset_embeddings.shape[1],
                k=k,
                niter=20,
                verbose=False,
                gpu=gpu_available,
                seed=42
            )
            kmeans.train(subset_embeddings)
            inertia.append(kmeans.obj[-1])
            pbar.set_postfix({'k': k, 'inertia': f'{kmeans.obj[-1]:.2e}'})
            pbar.update(1)
    
    print(f"✓ Elbow method completed")
    
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
    
    # Initialize FAISS K-Means with GPU support
    clusterer = faiss.Kmeans(
        d=normalized_embeddings.shape[1],
        k=optimal_k,
        niter=20,
        verbose=True,
        gpu=gpu_available,
        seed=42
    )
    
    print("\n⏳ Training K-Means (this may take a while)...")
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

for cluster_id in sorted(phrases_df['cluster'].unique()):
    cluster_data = phrases_df[phrases_df['cluster'] == cluster_id]
    
    total_phrases = len(cluster_data)
    hate_phrases = (cluster_data['label'] == 1).sum()
    hate_ratio = hate_phrases / total_phrases if total_phrases > 0 else 0
    label_variance = cluster_data['label'].var()
    sample_phrases = cluster_data['phrase'].unique()[:5].tolist()
    
    cluster_stats.append({
        'cluster_id': cluster_id,
        'size': total_phrases,
        'unique_phrases': cluster_data['phrase'].nunique(),
        'hate_count': hate_phrases,
        'non_hate_count': total_phrases - hate_phrases,
        'hate_ratio': hate_ratio,
        'label_variance': label_variance,
        'purity': max(hate_ratio, 1 - hate_ratio),
        'sample_phrases': sample_phrases
    })

cluster_stats_df = pd.DataFrame(cluster_stats)

print("\nCluster Statistics:")
print("=" * 80)
print(f"\n{cluster_stats_df[['cluster_id', 'size', 'unique_phrases', 'hate_ratio', 'label_variance']].to_string()}")

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
        'normalized_embeddings': normalized_embeddings,
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



