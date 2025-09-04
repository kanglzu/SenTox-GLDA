import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sentox_glda import SenToxGLDA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import os
import logging
import argparse
from datetime import datetime
import warnings
import random
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COLDataset(Dataset):
    """Dataset class for COLDataset (Chinese Online abusive Language dataset)"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize using the same tokenizer for both encoders
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SenToxGLDATrainer:
    """Trainer class for SenTox-GLDA model following paper specifications"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_logging()

    def setup_model(self):
        """Initialize model and tokenizer"""
        # Use BERT tokenizer as the primary tokenizer (compatible with both encoders)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

        # Initialize SenTox-GLDA model
        self.model = SenToxGLDA(
            sentiment_model_path=self.config['sentiment_model_path'],
            toxicity_model_path=self.config['toxicity_model_path'],
            num_classes=self.config['num_classes'],
            hidden_size=self.config['hidden_size'],
            num_heads=self.config['num_heads'],
            glda_downsample_ratio=self.config['glda_downsample_ratio'],
            dropout=self.config['dropout'],
            freeze_encoders=self.config['freeze_encoders']
        ).to(self.device)

        logger.info(f"Model initialized on {self.device}")

        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def setup_logging(self):
        """Setup logging directory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(self.config['output_dir'], f'sentox_glda_{timestamp}')
        os.makedirs(self.log_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def load_col_dataset(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load COLDataset following paper specifications"""
        try:
            df = pd.read_csv(data_path)
            # Assuming columns: 'text' and 'label' (0: normal, 1: abusive)
            texts = df['text'].astype(str).tolist()
            labels = df['label'].tolist()

            logger.info(f"Loaded {len(texts)} samples from COLDataset")

            # Log label distribution
            label_dist = pd.Series(labels).value_counts().to_dict()
            logger.info(f"Label distribution: {label_dist}")

            # Handle class imbalance
            unique_labels = np.unique(labels)
            self.class_weights = compute_class_weight(
                'balanced',
                classes=unique_labels,
                y=labels
            )
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
            logger.info(f"Class weights: {self.class_weights}")

            return texts, labels

        except Exception as e:
            logger.error(f"Error loading COLDataset: {e}")
            raise

    def create_data_loaders(self, texts: List[str], labels: List[int]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test data loaders with stratified split"""

        # First split: train+val / test (80% / 20%)
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Second split: train / val (80% / 20% of train_val)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels
        )

        # Create datasets
        train_dataset = COLDataset(train_texts, train_labels, self.tokenizer, self.config['max_length'])
        val_dataset = COLDataset(val_texts, val_labels, self.tokenizer, self.config['max_length'])
        test_dataset = COLDataset(test_texts, test_labels, self.tokenizer, self.config['max_length'])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

        logger.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            # Compute weighted loss
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        return total_loss / num_batches

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0

        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']

                # Calculate loss
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                # Get predictions
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class

        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')

        # Per-class F1 scores
        f1_per_class = f1_score(all_labels, all_predictions, average=None)

        # AUC score
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            auc = 0.0

        avg_loss = total_loss / len(val_loader)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'f1_class_0': f1_per_class[0] if len(f1_per_class) > 0 else 0,
            'f1_class_1': f1_per_class[1] if len(f1_per_class) > 1 else 0,
        }

        return metrics

    def train(self, data_path: str):
        """
        Main training loop following paper specifications:
        - AdamW optimizer (β1=0.9, β2=0.999, ε=1e-8)
        - Learning rate: 2e-5
        - Linear warmup (10% of total steps)
        - Weight decay: 0.01
        - Batch size: 32
        - Sequence length: 256
        - 25 epochs
        """

        # Load data
        texts, labels = self.load_col_dataset(data_path)
        train_loader, val_loader, test_loader = self.create_data_loaders(texts, labels)

        # Setup optimizer following paper specifications
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Setup scheduler
        total_steps = len(train_loader) * self.config['epochs']
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        best_macro_f1 = 0
        train_history = []

        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config['epochs']}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")

            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            logger.info(f"Val AUC: {val_metrics['auc']:.4f}")

            # Save training history
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            train_history.append(epoch_history)

            # Save best model
            if val_metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = val_metrics['macro_f1']
                self.save_model(os.path.join(self.log_dir, 'best_model'))
                logger.info(f"New best model saved! Macro-F1: {best_macro_f1:.4f}")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_dir = os.path.join(self.log_dir, f'checkpoint_epoch_{epoch + 1}')
                self.save_model(checkpoint_dir)

        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics = self.evaluate(test_loader)

        logger.info("=== Final Test Results ===")
        for metric_name, metric_value in test_metrics.items():
            logger.info(f"Test {metric_name}: {metric_value:.4f}")

        # Save training history and final results
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2)

        final_results = {
            'best_val_macro_f1': best_macro_f1,
            'test_metrics': test_metrics,
            'training_history': train_history
        }

        results_path = os.path.join(self.log_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info("Training completed!")
        logger.info(f"Best validation Macro-F1: {best_macro_f1:.4f}")
        logger.info(f"Results saved to: {self.log_dir}")

        return final_results

    def save_model(self, save_dir: str):
        """Save model and training state"""
        os.makedirs(save_dir, exist_ok=True)

        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))

        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)

        # Save config
        config_path = os.path.join(save_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to {save_dir}")

    def load_model(self, model_dir: str):
        """Load trained model"""
        model_path = os.path.join(model_dir, 'model.pt')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"Model loaded from {model_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train SenTox-GLDA model")

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to COLDataset CSV file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for models and logs')

    # Model arguments
    parser.add_argument('--sentiment_model_path', type=str, default='./models/bert_sentiment_encoder',
                        help='Path to trained BERT sentiment encoder')
    parser.add_argument('--toxicity_model_path', type=str, default='./models/roberta_toxicity_encoder',
                        help='Path to trained RoBERTa toxicity encoder')
    parser.add_argument('--freeze_encoders', action='store_true', help='Freeze pre-trained encoders')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')

    # Model architecture arguments
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--glda_downsample_ratio', type=int, default=4, help='GLDA downsampling ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function"""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create config dictionary
    config = {
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'sentiment_model_path': args.sentiment_model_path,
        'toxicity_model_path': args.toxicity_model_path,
        'num_classes': 2,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'glda_downsample_ratio': args.glda_downsample_ratio,
        'dropout': args.dropout,
        'freeze_encoders': args.freeze_encoders,
        'num_workers': args.num_workers,
        'seed': args.seed
    }

    # Log configuration
    logger.info("SenTox-GLDA Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Initialize trainer and start training
    trainer = SenToxGLDATrainer(config)
    results = trainer.train(args.data_path)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
