import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import logging
import json
import os
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToxicityDataset(Dataset):
    """Dataset class for ToxiCN toxicity detection"""

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

        # Tokenize
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


class RoBERTaToxicityTrainer:
    """RoBERTa trainer for toxicity detection as described in SenTox-GLDA paper"""

    def __init__(self, model_name: str = 'hfl/chinese-roberta-wwm-ext', num_labels: int = 2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)

        # Handle class imbalance (common in toxicity datasets)
        self.class_weights = None

        logger.info(f"Initialized RoBERTa model: {model_name}")
        logger.info(f"Device: {self.device}")

    def load_toxicn_data(self, data_path: str, augmented_data_path: Optional[str] = None) -> Tuple[
        List[str], List[int]]:
        """
        Load ToxiCN dataset and optionally augmented data
        According to paper: augmented ToxiCN dataset is used for training
        """
        try:
            # Load original ToxiCN data
            df = pd.read_csv(data_path)
            texts = df['text'].tolist()
            labels = df['label'].tolist()  # 0: non-toxic, 1: toxic

            logger.info(f"Loaded {len(texts)} samples from original ToxiCN dataset")

            # Load augmented data if provided
            if augmented_data_path and os.path.exists(augmented_data_path):
                aug_df = pd.read_csv(augmented_data_path)
                aug_texts = aug_df['text'].tolist()
                aug_labels = aug_df['label'].tolist()

                texts.extend(aug_texts)
                labels.extend(aug_labels)

                logger.info(f"Added {len(aug_texts)} augmented samples")
                logger.info(f"Total dataset size: {len(texts)}")

            # Log label distribution
            label_dist = pd.Series(labels).value_counts().to_dict()
            logger.info(f"Label distribution: {label_dist}")

            # Calculate class weights for imbalanced dataset
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
            logger.error(f"Error loading data: {e}")
            raise

    def create_data_loaders(self, texts: List[str], labels: List[int],
                            test_size: float = 0.2, batch_size: int = 32,
                            max_length: int = 256) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders with stratified split"""

        # Stratified split to maintain label distribution
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = ToxicityDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = ToxicityDataset(val_texts, val_labels, self.tokenizer, max_length)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch with class-weighted loss"""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Apply class weighting if available
            loss = outputs.loss
            if self.class_weights is not None:
                # Manual weighted loss calculation
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits, labels)

            total_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance with toxicity-specific metrics"""
        self.model.eval()
        predictions = []
        true_labels = []
        probabilities = []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Calculate weighted loss if class weights are available
                if self.class_weights is not None:
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                    loss = loss_fct(logits, labels)
                else:
                    loss = outputs.loss

                total_loss += loss.item()

                # Get predictions and probabilities
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of toxic class

        # Calculate comprehensive metrics for toxicity detection
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        weighted_f1 = f1_score(true_labels, predictions, average='weighted')

        # Calculate per-class F1 scores
        f1_scores = f1_score(true_labels, predictions, average=None)
        non_toxic_f1 = f1_scores[0] if len(f1_scores) > 0 else 0
        toxic_f1 = f1_scores[1] if len(f1_scores) > 1 else 0

        # AUC score
        try:
            auc_score = roc_auc_score(true_labels, probabilities)
        except ValueError:
            auc_score = 0.0

        avg_loss = total_loss / len(val_loader)

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'non_toxic_f1': non_toxic_f1,
            'toxic_f1': toxic_f1,
            'auc': auc_score,
            'loss': avg_loss
        }

    def train(self, data_path: str,
              augmented_data_path: Optional[str] = None,
              epochs: int = 25,
              batch_size: int = 32,
              learning_rate: float = 2e-5,
              max_length: int = 256,
              save_dir: str = "./roberta_toxicity_model"):
        """
        Train RoBERTa for toxicity detection following paper specifications:
        - Uses augmented ToxiCN dataset
        - AdamW optimizer (β1=0.9, β2=0.999)
        - Learning rate: 2e-5
        - Linear warmup
        - Batch size: 32
        - Sequence length: 256
        - 25 epochs
        """

        # Load data
        texts, labels = self.load_toxicn_data(data_path, augmented_data_path)
        train_loader, val_loader = self.create_data_loaders(
            texts, labels, batch_size=batch_size, max_length=max_length
        )

        # Setup optimizer and scheduler (matching paper specifications)
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps
        )

        # Training loop
        best_macro_f1 = 0
        train_history = []

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            logger.info(f"Val Toxic-F1: {val_metrics['toxic_f1']:.4f}")
            logger.info(f"Val AUC: {val_metrics['auc']:.4f}")

            # Save history
            train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_weighted_f1': val_metrics['weighted_f1'],
                'val_non_toxic_f1': val_metrics['non_toxic_f1'],
                'val_toxic_f1': val_metrics['toxic_f1'],
                'val_auc': val_metrics['auc']
            })

            # Save best model based on macro F1 (as mentioned in paper)
            if val_metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = val_metrics['macro_f1']
                self.save_model(save_dir)
                logger.info(f"New best model saved with Macro-F1: {best_macro_f1:.4f}")

        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2)

        logger.info("RoBERTa toxicity training completed!")
        logger.info(f"Best Macro-F1: {best_macro_f1:.4f}")

        return train_history

    def cross_validate(self, data_path: str, augmented_data_path: Optional[str] = None,
                       k_folds: int = 5, **train_kwargs) -> List[Dict]:
        """Perform k-fold cross validation for robust evaluation"""
        texts, labels = self.load_toxicn_data(data_path, augmented_data_path)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            logger.info(f"Cross-validation fold {fold + 1}/{k_folds}")

            # Split data
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            # Create data loaders
            train_dataset = ToxicityDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = ToxicityDataset(val_texts, val_labels, self.tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Reinitialize model for each fold
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            ).to(self.device)

            # Train on fold
            history = self.train_fold(train_loader, val_loader, **train_kwargs)
            cv_results.append(history)

        return cv_results

    def save_model(self, save_dir: str):
        """Save model and tokenizer"""
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save additional metadata
        metadata = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'class_weights': self.class_weights.tolist() if self.class_weights is not None else None
        }
        with open(os.path.join(save_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {save_dir}")

    def load_model(self, model_dir: str):
        """Load trained model"""
        self.model = RobertaForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)

        # Load metadata if available
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if metadata['class_weights']:
                    self.class_weights = torch.tensor(metadata['class_weights']).to(self.device)

        logger.info(f"Model loaded from {model_dir}")


def main():
    """Main training function for RoBERTa toxicity encoder"""
    # Configuration matching paper specifications
    config = {
        'model_name': 'hfl/chinese-roberta-wwm-ext',  # Chinese RoBERTa
        'data_path': './data/toxicn_dataset.csv',  # Original ToxiCN dataset
        'augmented_data_path': './data/toxicn_augmented.csv',  # Augmented dataset
        'epochs': 25,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'max_length': 256,
        'save_dir': './models/roberta_toxicity_encoder'
    }

    # Initialize trainer
    trainer = RoBERTaToxicityTrainer(
        model_name=config['model_name'],
        num_labels=2  # Binary toxicity classification
    )

    # Train model
    history = trainer.train(
        data_path=config['data_path'],
        augmented_data_path=config['augmented_data_path'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        max_length=config['max_length'],
        save_dir=config['save_dir']
    )

    print("RoBERTa toxicity encoder training completed!")


if __name__ == "__main__":
    main()
