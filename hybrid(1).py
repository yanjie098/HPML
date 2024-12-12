import os
import time
import torch
import wandb
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import deepspeed
import logging
from tqdm import tqdm

class HybridParallelismTrainer:
    def __init__(
        self,
        model_name='gpt2',
        num_labels=4,
        learning_rate=5e-5,
        batch_size=128,
        epochs=3
    ):
        """
        Initialize trainer with advanced hybrid parallelism

        Args:
            model_name (str): Pretrained model name
            num_labels (int): Number of classes in dataset
            learning_rate (float): Initial learning rate
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
        """
        # Ensure environment is set up
        self.setup_environment()

        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Prepare dataset
        self.dataset = self.prepare_dataset()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_environment(self):
        """
        Ensure distributed environment is properly set up
        """
        # Set default values if not already set
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'

    def prepare_dataset(self):
        """
        Prepare AG News dataset with tokenization

        Returns:
            dict: Tokenized and prepared datasets
        """
        # Load AG News dataset
        dataset = load_dataset('ag_news')

        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )

        # Prepare dataset
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            num_proc=os.cpu_count()
        )

        # Convert to PyTorch format
        tokenized_datasets.set_format(
            'torch',
            columns=['input_ids', 'attention_mask', 'label']
        )

        return tokenized_datasets

    def create_dataloader(self, dataset, batch_size=4):
        """
        Create a DataLoader with distributed sampling

        Args:
            dataset (Dataset): Input dataset
            batch_size (int): Batch size

        Returns:
            DataLoader: Distributed data loader
        """
        # Create sampler
        sampler = RandomSampler(dataset)

        # Create DataLoader
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size
        )

    def train(self, local_rank=0):
        """
        Training method with fallback for single-device training

        Args:
            local_rank (int): Local GPU rank
        """
        # Initialize wandb
        if local_rank == 0:
            wandb.login(key='e56a4a4fe9514526421692a0b66ad06d01dd431b')
            wandb.init(
                project="gpt2-ag-news-classification-damn",
                config={
                    "model": self.model_name,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs
                }
            )

        # Prepare model
        model = self.model

        # Move model to device
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        self.logger.info(f"Starting training on device: {device}")

        # Prepare datasets
        train_dataset = self.dataset['train']
        test_dataset = self.dataset['test']

        # Create data loaders
        train_dataloader = self.create_dataloader(train_dataset, self.batch_size)
        test_dataloader = self.create_dataloader(test_dataset, 1)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        # Training loop
        model.train()
        total_steps = 0
        for epoch in range(self.epochs):
            total_loss = 0
            start_time = time.time()
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")):
                # Prepare batch
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()

                # Accumulate loss
                total_loss += loss.item()
                total_steps += 1

                # Log batch details
                self.logger.info(f"Epoch {epoch+1}, Batch loss: {loss.item()}")

                # Log to wandb every 100 steps
                if local_rank == 0 and total_steps % 100 == 0:
                    elapsed_time = time.time() - start_time
                    steps_per_second = total_steps / elapsed_time
                    samples_per_second = total_steps * self.batch_size / elapsed_time
                    wandb.log({
                        'epoch': epoch,
                        'step': total_steps,
                        'loss': loss.item(),
                        'steps_per_second': steps_per_second,
                        'samples_per_second': samples_per_second
                    })

            # Print average loss per epoch
            avg_loss = total_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss}")
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss}")

        # Evaluate model
        if local_rank == 0:
            self.evaluate_model(model, test_dataloader, device)

    def evaluate_model(self, model, dataloader, device):
        """
        Evaluate model performance

        Args:
            model (nn.Module): Trained model
            dataloader (DataLoader): Test data loader
            device (torch.device): Computing device
        """
        model.eval()
        total_correct = 0
        total_samples = 0
        total_inference_time = 0

        self.logger.info("Starting evaluation")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Prepare batch
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # Measure inference time
                start_time = time.time()

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Calculate inference time
                inference_time = time.time() - start_time
                total_inference_time += inference_time

                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        # Calculate metrics
        accuracy = total_correct / total_samples
        avg_inference_time = total_inference_time / len(dataloader)

        # Log to wandb
        wandb.log({
            'test_accuracy': accuracy,
            'avg_inference_time': avg_inference_time
        })

        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"Average Inference Time: {avg_inference_time:.4f} seconds")

        # Print metrics
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Average Inference Time: {avg_inference_time:.4f} seconds")

def main():
    """
    Main training function with flexible device handling
    """
    # Determine available devices
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if torch.cuda.is_available():
        # Multi-GPU training
        num_gpus = torch.cuda.device_count()
        logger.info(f"Training on {num_gpus} GPUs")
        print(f"Training on {num_gpus} GPUs")

        # Single-process training (simpler for Colab/Jupyter)
        trainer = HybridParallelismTrainer()
        trainer.train()
    else:
        # CPU fallback
        logger.info("No GPU available. Training on CPU.")
        print("No GPU available. Training on CPU.")
        trainer = HybridParallelismTrainer()
        trainer.train()

if __name__ == '__main__':
    main()