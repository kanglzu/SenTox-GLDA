import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import logging
import json
import os
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """Dataset class for Weibo-100k sentiment analysis"""

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


class BERTSentimentTrainer:
    """BERT trainer for sentiment analysis as described in SenTox-GLDA paper"""

    def __init__(self, model_name: str = 'bert-base-chinese', num_labels: int = 2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)

        logger.info(f"Initialized BERT model: {model_name}")
        logger.info(f"Device: {self.device}")

    def load_weibo_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load Weibo-100k dataset"""
        try:
            df = pd.read_csv(data_path)
            # Assuming columns: 'text' and 'label' (0: negative, 1: positive)
            texts = df['text'].tolist()
            labels = df['label'].tolist()

            logger.info(f"Loaded {len(texts)} samples from {data_path}")
            logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

            return texts, labels

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def create_data_loaders(self, texts: List[str], labels: List[int],
                            test_size: float = 0.2, batch_size: int = 32,
                            max_length: int = 256) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, max_length)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch"""
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

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        predictions = []
        true_labels = []
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

                total_loss += outputs.loss.item()

                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        avg_loss = total_loss / len(val_loader)

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'loss': avg_loss
        }

    def train(self, data_path: str,
              epochs: int = 25,
              batch_size: int = 32,
              learning_rate: float = 2e-5,
              max_length: int = 256,
              save_dir: str = "./bert_sentiment_model"):
        """
        Train BERT for sentiment analysis following paper specifications:
        - AdamW optimizer (β1=0.9, β2=0.999)
        - Learning rate: 2e-5
        - Linear warmup
        - Batch size: 32
        - Sequence length: 256
        - 25 epochs
        """

        # Load data
        texts, labels = self.load_weibo_data(data_path)
        train_loader, val_loader = self.create_data_loaders(
            texts, labels, batch_size=batch_size, max_length=max_length
        )

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Training loop
        best_f1 = 0
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

            # Save history
            train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': val_metrics['macro_f1']
            })

            # Save best model
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                self.save_model(save_dir)
                logger.info(f"New best model saved with Macro-F1: {best_f1:.4f}")

        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2)

        logger.info("Training completed!")
        logger.info(f"Best Macro-F1: {best_f1:.4f}")

        return train_history

    def save_model(self, save_dir: str):
        """Save model and tokenizer"""
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")

    def load_model(self, model_dir: str):
        """Load trained model"""
        self.model = BertForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        logger.info(f"Model loaded from {model_dir}")


def main():
    """Main training function"""
    # Configuration matching paper specifications
    config = {
        'model_name': 'bert-base-chinese',
        'data_path': './data/weibo_100k.csv',  # Path to Weibo-100k dataset
        'epochs': 25,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'max_length': 256,
        'save_dir': './models/bert_sentiment_encoder'
    }

    # Initialize trainer
    trainer = BERTSentimentTrainer(
        model_name=config['model_name'],
        num_labels=2  # Binary sentiment classification
    )

    # Train model
    history = trainer.train(
        data_path=config['data_path'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        max_length=config['max_length'],
        save_dir=config['save_dir']
    )

    print("BERT sentiment encoder training completed!")


if __name__ == "__main__":
    main()
