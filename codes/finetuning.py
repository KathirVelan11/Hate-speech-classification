vanako

import pandas as pd
import numpy as np
import pickle
import warnings
import os
import torch
import logging
from datetime import datetime
warnings.filterwarnings('ignore')

# Hugging Face datasets for efficient loading
from datasets import load_dataset, Dataset, ClassLabel

# KeyBERT for phrase extraction - REMOVED from here, will load in phrase_extraction.py
# Not needed for fine-tuning, only for phrase extraction

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

# LoRA and PEFT for efficient fine-tuning
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Utilities
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm


# ============================================================================
# STEP 1: INITIALIZE MODELS AND CUSTOM STOP WORDS
# ============================================================================

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("=" * 80)
print("EXPLAINABLE HATE SPEECH CLASSIFICATION - FINE-TUNING")
print("=" * 80)
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("Running on CPU (CUDA not available)")

print(f"PEFT (LoRA) Available: {PEFT_AVAILABLE}")
if not PEFT_AVAILABLE:
    print("Warning: PEFT library not found. Will use standard fine-tuning if needed.")

print("\n[1/3] Loading models and preparing stop words...")

# KeyBERT is not used in fine-tuning, only in phrase extraction
# Skip loading it here to save memory

# Initialize embedding model with GPU support
try:
    print("\n[Loading embedding model 'paraphrase-mpnet-base-v2' from HuggingFace...]")
    print(f"Device: {device}")
    print("[This may take 2-5 minutes on first run...]", flush=True)
    import sys
    sys.stdout.flush()
    
    embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)
    print(f"✓ Embedding model loaded successfully on {device}", flush=True)
except Exception as e:
    print(f"ERROR: Failed to load embedding model on {device}: {e}")
    print("Warning: Attempting to load on CPU...")
    try:
        embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
        device = 'cpu'
        print("Embedding model loaded on CPU")
    except Exception as e2:
        print(f"ERROR: Failed to load embedding model on CPU: {e2}")
        raise

print("Models loaded successfully!")
print(f"  - Embedding Model: paraphrase-mpnet-base-v2 on {device}")

# ============================================================================
# Custom Stop Words for Hate Speech Detection
# ============================================================================

# Standard English stop words
standard_stop_words = set(ENGLISH_STOP_WORDS)

# Words we MUST KEEP for hate speech detection (remove from stop words)
hate_relevant_words = {
    # Emotional/sentiment words
    'hate', 'love', 'like', 'dislike', 'fear',
    
    # Negations (critical!)
    'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', "n't",
    
    # Intensifiers
    'very', 'really', 'so', 'too', 'quite', 'extremely', 'totally', 'completely',
    
    # Modal verbs (intent/obligation)
    'should', 'must', 'ought', 'need', 'have', 'has', 'had',
    
    # Quantifiers (generalization indicators)
    'all', 'every', 'each', 'any', 'some', 'few', 'many', 'most', 'several',
    
    # Pronouns (target identification)
    'they', 'them', 'their', 'those', 'these', 'we', 'us', 'our',
    
    # Directional/spatial (important for exclusion phrases)
    'back', 'away', 'out', 'off', 'down',
    
    # Comparative/superlative
    'more', 'less', 'better', 'worse', 'best', 'worst',
    
    # Other contextually important words
    'only', 'just', 'still', 'even', 'again', 'against'
}

# Create custom stop word list
custom_stop_words = standard_stop_words - hate_relevant_words

# Add domain-specific stop words (optional - words that add no value)
additional_stop_words = {
    'said', 'says', 'saying', 'say',  # Reporting verbs
    'would', 'could', 'might', 'may',  # Weak modals
}

custom_stop_words = custom_stop_words.union(additional_stop_words)

print(f"\nCustom stop word list created!")
print(f"  - Standard stop words: {len(standard_stop_words)}")
print(f"  - Custom stop words: {len(custom_stop_words)}")
print(f"  - Kept {len(hate_relevant_words)} hate-relevant words")
print(f"\nSample kept words: {list(hate_relevant_words)[:10]}")

# Save custom stop words for use in phrase extraction
with open('custom_stop_words.pkl', 'wb') as f:
    pickle.dump(custom_stop_words, f)
print("Saved custom stop words to custom_stop_words.pkl")


# ============================================================================
# STEP 2: LOAD AND PREPARE DATASET
# ============================================================================

print(f"\n[2/3] Loading dataset...")

# Load dataset using Hugging Face datasets for memory efficiency
try:
    print("[Loading HateSpeechDatasetBalanced.csv...]", flush=True)
    dataset = load_dataset('csv', data_files='HateSpeechDatasetBalanced.csv', split='train')
    print(f"✓ Dataset loaded: {len(dataset)} samples", flush=True)
except Exception as e:
    print(f"ERROR: Failed to load dataset: {e}")
    raise

# Convert label column to ClassLabel for stratified split
if not isinstance(dataset.features['label'], ClassLabel):
    unique_labels = sorted(set(dataset['label']))
    dataset = dataset.cast_column('label', ClassLabel(names=[str(x) for x in unique_labels]))

# Now split with stratification
dataset_split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
train_dataset = dataset_split['train']
val_dataset = dataset_split['test']

print(f"\nDataset split for fine-tuning:")
print(f"  - Training samples: {len(train_dataset)}")
print(f"  - Validation samples: {len(val_dataset)}")

# Save full dataset for phrase extraction
dataset.save_to_disk('full_dataset')
print("Saved full dataset to full_dataset/")

# del df_sample  # Free memory


# ============================================================================
# STEP 2.5: FINE-TUNE EMBEDDING MODEL WITH LoRA (MEMORY-OPTIMIZED)
# ============================================================================

print(f"\n[3/3] Fine-tuning embedding model with LoRA (Memory-Optimized)...")

# Setup dedicated logger for fine-tuning with detailed file logging
finetune_logger = logging.getLogger('LoRA_FineTuning')
finetune_logger.setLevel(logging.DEBUG)

# Create file handler for detailed fine-tuning logs
fh = logging.FileHandler(f'lora_finetuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
fh.setLevel(logging.DEBUG)

# Create console handler with higher level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers
finetune_logger.addHandler(fh)
finetune_logger.addHandler(ch)

finetune_logger.info("\n" + "="*80)
finetune_logger.info("LORA FINE-TUNING INITIALIZATION (MEMORY-OPTIMIZED)")
finetune_logger.info("="*80)
finetune_logger.info(f"Device: {device}")
finetune_logger.info(f"Base Model: paraphrase-mpnet-base-v2")
finetune_logger.info(f"Output Path: ./hate_speech_fine_tuned_model")

# Print current model parameters
total_params = sum(p.numel() for p in embedding_model.parameters())
trainable_params = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
finetune_logger.info(f"\nModel Parameters:")
finetune_logger.info(f"  Total: {total_params:,}")
finetune_logger.info(f"  Trainable: {trainable_params:,}")
finetune_logger.info(f"  Memory: ~{total_params * 4 / 1024**3:.2f} GB (FP32)")

# Apply LoRA if PEFT is available
if PEFT_AVAILABLE:
    try:
        finetune_logger.info("\n" + "-"*80)
        finetune_logger.info("APPLYING LORA ADAPTERS")
        finetune_logger.info("-"*80)
        
        # Configure LoRA with memory-efficient settings for RTX 4060
        lora_config = LoraConfig(
            r=2,  # Very low rank due to 4GB GPU memory
            lora_alpha=4,
            target_modules=["query", "value"],  # Only attention layers
            lora_dropout=0.2,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        finetune_logger.debug(f"LoRA Config: {lora_config}")
        
        # Access the underlying transformer model
        base_transformer = embedding_model[0].auto_model
        finetune_logger.debug(f"Base transformer type: {type(base_transformer)}")
        
        # Apply LoRA
        finetune_logger.info("Applying LoRA to model...")
        base_transformer = get_peft_model(base_transformer, lora_config)
        embedding_model[0].auto_model = base_transformer
        
        # Print LoRA parameter counts
        lora_params = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
        finetune_logger.info(f"\n✓ LoRA Applied Successfully!")
        finetune_logger.info(f"  Trainable params: {lora_params:,}")
        finetune_logger.info(f"  Reduction: {(1 - lora_params/trainable_params)*100:.2f}%")
        finetune_logger.info(f"  LoRA rank: {lora_config.r}")
        finetune_logger.info(f"  LoRA alpha: {lora_config.lora_alpha}")
        finetune_logger.info(f"  LoRA dropout: {lora_config.lora_dropout}")
        
    except Exception as e:
        finetune_logger.error(f"LoRA application failed: {e}", exc_info=True)
        finetune_logger.warning("Falling back to standard fine-tuning...")
else:
    finetune_logger.warning("PEFT library not available, using standard fine-tuning")

# Prepare training data: create sentence pairs
finetune_logger.info("\n" + "-"*80)
finetune_logger.info("PREPARING TRAINING DATA")
finetune_logger.info("-"*80)
train_examples = []

# Convert to pandas for easier manipulation
try:
    train_df = train_dataset.to_pandas()
    finetune_logger.debug(f"Training data shape: {train_df.shape}")
except Exception as e:
    finetune_logger.error(f"Failed to convert training data to pandas: {e}")
    raise

# Create positive pairs (same label) and negative pairs (different labels)
hate_texts = train_df[train_df['label'] == 1]['text'].tolist()
non_hate_texts = train_df[train_df['label'] == 0]['text'].tolist()

# Positive pairs (same class) - score 1.0
# Reduce pairs to fit in memory for RTX 4060 (4GB)
max_pairs = min(len(hate_texts), len(non_hate_texts), 2000)

for i in range(min(max_pairs, len(hate_texts) - 1)):
    train_examples.append(InputExample(texts=[hate_texts[i], hate_texts[i+1]], label=1.0))

for i in range(min(max_pairs, len(non_hate_texts) - 1)):
    train_examples.append(InputExample(texts=[non_hate_texts[i], non_hate_texts[i+1]], label=1.0))

# Negative pairs (different class) - score 0.0
for i in range(min(max_pairs, len(hate_texts))):
    idx = i % len(non_hate_texts)
    train_examples.append(InputExample(texts=[hate_texts[i], non_hate_texts[idx]], label=0.0))

finetune_logger.info(f"✓ Created {len(train_examples)} training pairs")
finetune_logger.debug(f"  - Positive pairs: {len([e for e in train_examples if e.label == 1.0])}")
finetune_logger.debug(f"  - Negative pairs: {len([e for e in train_examples if e.label == 0.0])}")

# Create DataLoader with very small batch size for RTX 4060
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)  # Reduced from 16
finetune_logger.info(f"✓ DataLoader created with batch_size=8")

# Define loss
train_loss = losses.CosineSimilarityLoss(embedding_model)
finetune_logger.info(f"✓ Loss function: CosineSimilarityLoss")

# Prepare validation data
try:
    finetune_logger.info("Preparing validation data...")
    val_df = val_dataset.to_pandas()
    finetune_logger.debug(f"Validation data shape: {val_df.shape}")
except Exception as e:
    finetune_logger.error(f"Failed to prepare validation data: {e}")
    raise

# Create evaluator
sentences1 = []
sentences2 = []
scores = []

# Create validation pairs
val_hate = val_df[val_df['label'] == 1]['text'].tolist()
val_non_hate = val_df[val_df['label'] == 0]['text'].tolist()

max_val_pairs = min(len(val_hate), len(val_non_hate), 500)

for i in range(min(max_val_pairs, len(val_hate) - 1)):
    sentences1.append(val_hate[i])
    sentences2.append(val_hate[i+1])
    scores.append(1.0)

for i in range(min(max_val_pairs, len(val_non_hate) - 1)):
    sentences1.append(val_non_hate[i])
    sentences2.append(val_non_hate[i+1])
    scores.append(1.0)

for i in range(min(max_val_pairs, len(val_hate))):
    idx = i % len(val_non_hate)
    sentences1.append(val_hate[i])
    sentences2.append(val_non_hate[idx])
    scores.append(0.0)

evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

finetune_logger.info(f"✓ Created {len(scores)} validation pairs")

# Train model
finetune_logger.info("\n" + "-"*80)
finetune_logger.info("STARTING FINE-TUNING")
finetune_logger.info("-"*80)
warmup_steps = max(10, int(len(train_dataloader) * 0.05))
finetune_logger.info(f"Epochs: 2 (reduced for memory efficiency)")
finetune_logger.info(f"Batch size: 8")
finetune_logger.info(f"Warmup steps: {warmup_steps}")
finetune_logger.info(f"Evaluation frequency: Every {len(train_dataloader) // 2} steps")
finetune_logger.info(f"Total training steps: {len(train_dataloader) * 2}")

try:
    finetune_logger.info("Training in progress...")
    embedding_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=2,  # Reduced from 3 for memory
        warmup_steps=warmup_steps,
        evaluation_steps=max(10, len(train_dataloader) // 3),
        output_path='./hate_speech_fine_tuned_model',
        save_best_model=True,
        show_progress_bar=True
    )
    
    finetune_logger.info("\n" + "="*80)
    finetune_logger.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
    finetune_logger.info("="*80)
    finetune_logger.info(f"Model saved to: ./hate_speech_fine_tuned_model")
except Exception as e:
    finetune_logger.error(f"Fine-tuning failed: {e}", exc_info=True)
    raise

# Clean up
del train_df, val_df, train_examples, train_dataloader
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Save model info for next steps
model_info = {
    'model_path': './hate_speech_fine_tuned_model',
    'base_model': 'paraphrase-mpnet-base-v2',
    'fine_tuned': True,
    'lora_applied': PEFT_AVAILABLE,
    'device': device
}

with open('embedding_model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ Saved embedding model info to embedding_model_info.pkl")
    """
    Fine-tune SentenceTransformer with LoRA adapters for hate speech domain.
    
    Parameters:
    - base_model_name: Name of the base model
    - train_data: Training dataset
    - val_data: Validation dataset
    - output_path: Path to save fine-tuned model
    - device: Device to use for training ('cuda' or 'cpu')
    """
    finetune_logger.info("\n" + "="*80)
    finetune_logger.info("LORA FINE-TUNING INITIALIZATION")
    finetune_logger.info("="*80)
    finetune_logger.info(f"Device: {device}")
    finetune_logger.info(f"Base Model: {base_model_name}")
    finetune_logger.info(f"Output Path: {output_path}")
    
    # Load base model
    try:
        finetune_logger.info("Loading base model...")
        model = SentenceTransformer(base_model_name, device=device)
        finetune_logger.info(f"✓ Base model loaded successfully on {device}")
    except Exception as e:
        finetune_logger.error(f"Failed to load base model on {device}: {e}")
        if device == 'cuda':
            finetune_logger.warning("Attempting fallback to CPU...")
            try:
                model = SentenceTransformer(base_model_name, device='cpu')
                device = 'cpu'
                finetune_logger.info("✓ Base model loaded on CPU")
            except Exception as e2:
                finetune_logger.critical(f"Failed to load model on CPU: {e2}")
                raise
        else:
            raise
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    finetune_logger.info(f"\nModel Parameters:")
    finetune_logger.info(f"  Total: {total_params:,}")
    finetune_logger.info(f"  Trainable: {trainable_params:,}")
    finetune_logger.info(f"  Memory: ~{total_params * 4 / 1024**3:.2f} GB (FP32)")
    
    # Apply LoRA if PEFT is available
    if PEFT_AVAILABLE:
        try:
            finetune_logger.info("\n" + "-"*80)
            finetune_logger.info("APPLYING LORA ADAPTERS")
            finetune_logger.info("-"*80)
            
            # Configure LoRA with heavy regularization
            lora_config = LoraConfig(
                r=4,  # Low rank for heavy regularization
                lora_alpha=8,
                target_modules=["query", "value"],  # Only attention layers
                lora_dropout=0.2,  # High dropout for regularization
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            finetune_logger.debug(f"LoRA Config: {lora_config}")
            
            # Access the underlying transformer model
            base_transformer = model[0].auto_model
            finetune_logger.debug(f"Base transformer type: {type(base_transformer)}")
            
            # Apply LoRA
            finetune_logger.info("Applying LoRA to model...")
            base_transformer = get_peft_model(base_transformer, lora_config)
            model[0].auto_model = base_transformer
            
            # Print LoRA parameter counts
            lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            finetune_logger.info(f"\n✓ LoRA Applied Successfully!")
            finetune_logger.info(f"  Trainable params: {lora_params:,}")
            finetune_logger.info(f"  Reduction: {(1 - lora_params/trainable_params)*100:.2f}%")
            finetune_logger.info(f"  LoRA rank: {lora_config.r}")
            finetune_logger.info(f"  LoRA alpha: {lora_config.lora_alpha}")
            finetune_logger.info(f"  LoRA dropout: {lora_config.lora_dropout}")
            finetune_logger.info(f"  Target modules: {lora_config.target_modules}")
            
        except Exception as e:
            finetune_logger.error(f"LoRA application failed: {e}", exc_info=True)
            finetune_logger.warning("Falling back to standard fine-tuning...")
    else:
        finetune_logger.warning("PEFT library not available, using standard fine-tuning")
    
    # Prepare training data: create sentence pairs
    finetune_logger.info("\n" + "-"*80)
    finetune_logger.info("PREPARING TRAINING DATA")
    finetune_logger.info("-"*80)
    train_examples = []
    
    # Convert to pandas for easier manipulation
    try:
        train_df = train_data.to_pandas()
        finetune_logger.debug(f"Training data shape: {train_df.shape}")
    except Exception as e:
        finetune_logger.error(f"Failed to convert training data to pandas: {e}")
        raise
    
    # Create positive pairs (same label) and negative pairs (different labels)
    hate_texts = train_df[train_df['label'] == 1]['text'].tolist()
    non_hate_texts = train_df[train_df['label'] == 0]['text'].tolist()
    
    # Positive pairs (same class) - score 1.0
    max_pairs = min(len(hate_texts), len(non_hate_texts), 5000)  # Limit for memory
    
    for i in range(min(max_pairs, len(hate_texts) - 1)):
        train_examples.append(InputExample(texts=[hate_texts[i], hate_texts[i+1]], label=1.0))
    
    for i in range(min(max_pairs, len(non_hate_texts) - 1)):
        train_examples.append(InputExample(texts=[non_hate_texts[i], non_hate_texts[i+1]], label=1.0))
    
    # Negative pairs (different class) - score 0.0
    for i in range(min(max_pairs, len(hate_texts))):
        idx = i % len(non_hate_texts)
        train_examples.append(InputExample(texts=[hate_texts[i], non_hate_texts[idx]], label=0.0))
    
    finetune_logger.info(f"✓ Created {len(train_examples)} training pairs")
    finetune_logger.debug(f"  - Positive pairs: {len([e for e in train_examples if e.label == 1.0])}")
    finetune_logger.debug(f"  - Negative pairs: {len([e for e in train_examples if e.label == 0.0])}")
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    finetune_logger.info(f"DataLoader created with batch_size=16")
    
    # Define loss
    train_loss = losses.CosineSimilarityLoss(model)
    finetune_logger.info(f"Loss function: CosineSimilarityLoss")
    
    # Prepare validation data
    try:
        finetune_logger.info("Preparing validation data...")
        val_df = val_data.to_pandas()
        val_texts = val_df['text'].tolist()
        val_labels = val_df['label'].tolist()
        finetune_logger.debug(f"Validation data shape: {val_df.shape}")
    except Exception as e:
        finetune_logger.error(f"Failed to prepare validation data: {e}")
        raise
    
    # Create evaluator
    sentences1 = []
    sentences2 = []
    scores = []
    
    # Create validation pairs
    val_hate = val_df[val_df['label'] == 1]['text'].tolist()
    val_non_hate = val_df[val_df['label'] == 0]['text'].tolist()
    
    max_val_pairs = min(len(val_hate), len(val_non_hate), 1000)
    
    for i in range(min(max_val_pairs, len(val_hate) - 1)):
        sentences1.append(val_hate[i])
        sentences2.append(val_hate[i+1])
        scores.append(1.0)
    
    for i in range(min(max_val_pairs, len(val_non_hate) - 1)):
        sentences1.append(val_non_hate[i])
        sentences2.append(val_non_hate[i+1])
        scores.append(1.0)
    
    for i in range(min(max_val_pairs, len(val_hate))):
        idx = i % len(val_non_hate)
        sentences1.append(val_hate[i])
        sentences2.append(val_non_hate[idx])
        scores.append(0.0)
    
    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    
    finetune_logger.info(f"✓ Created {len(scores)} validation pairs")
    finetune_logger.debug(f"  - Positive validation pairs: {sum(1 for s in scores if s == 1.0)}")
    finetune_logger.debug(f"  - Negative validation pairs: {sum(1 for s in scores if s == 0.0)}")
    
    # Train model
    finetune_logger.info("\n" + "-"*80)
    finetune_logger.info("STARTING FINE-TUNING")
    finetune_logger.info("-"*80)
    warmup_steps = int(len(train_dataloader) * 0.1)
    finetune_logger.info(f"Epochs: 1")
    finetune_logger.info(f"Warmup steps: {warmup_steps}")
    finetune_logger.info(f"Evaluation steps: {len(train_dataloader) // 2}")
    finetune_logger.info(f"Total training steps: {len(train_dataloader)}")
    
    try:
        finetune_logger.info("Training in progress...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=3,
            warmup_steps=warmup_steps,
            evaluation_steps=len(train_dataloader) // 2,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True
        )
        
        finetune_logger.info("\n" + "="*80)
        finetune_logger.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
        finetune_logger.info("="*80)
        finetune_logger.info(f"Model saved to: {output_path}")
    except Exception as e:
        finetune_logger.error(f"Fine-tuning failed: {e}", exc_info=True)
        raise
    
    # Load best model
    try:
        finetune_logger.info("Loading best model...")
        model = SentenceTransformer(output_path, device=device)
        finetune_logger.info(f"✓ Best model loaded from {output_path}")
    except Exception as e:
        finetune_logger.error(f"Failed to load best model: {e}")
        raise
    
    # Clean up
    del train_df, val_df, train_examples, train_dataloader
    
    return model

# Fine-tune the model
try:
    embedding_model = fine_tune_with_lora(
        'paraphrase-mpnet-base-v2',
        train_dataset,
        val_dataset,
        output_path='./hate_speech_fine_tuned_model',
        device=device
    )
    print("\nUsing fine-tuned model for embeddings!")
    
    # Save model info for next steps
    model_info = {
        'model_path': './hate_speech_fine_tuned_model',
        'base_model': 'paraphrase-mpnet-base-v2',
        'fine_tuned': True,
        'lora_applied': PEFT_AVAILABLE,
        'device': device
    }
    
    with open('embedding_model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    print("Saved embedding model info to embedding_model_info.pkl")
    
except Exception as e:
    print("Warning:", f"\nFine-tuning failed: {e}")
    print("Falling back to base model...")
    try:
        embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)
        
        # Save fallback model info
        model_info = {
            'model_path': 'paraphrase-mpnet-base-v2',
            'base_model': 'paraphrase-mpnet-base-v2',
            'fine_tuned': False,
            'lora_applied': False,
            'device': device
        }
        
        with open('embedding_model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        print("Saved base model info to embedding_model_info.pkl")
        
    except Exception as e2:
        print("ERROR:", f"Failed to load base model: {e2}")
        raise


print("\n" + "=" * 80)
print("FINE-TUNING COMPLETE!")
print("=" * 80)
print("\nSaved files:")
print("  - Fine-tuned model: ./hate_speech_fine_tuned_model/")
print("  - Model info: embedding_model_info.pkl")
print("  - Custom stop words: custom_stop_words.pkl")
print("  - Full dataset: full_dataset/")
print("\nNext step: Run phrase_extraction.py")
