import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import our modules
from dataset import create_memory_dataloaders, MemoryDataset
from model import create_memory_model, HumanMemoryNet

class MemoryLoss(nn.Module):
    """
    Custom loss function that combines multiple objectives:
    - Memory type classification
    - Memory strength prediction
    - Sequence reconstruction
    - Memory consistency (encourages stable memory representations)
    """
    
    def __init__(self, 
                 type_weight: float = 1.0,
                 strength_weight: float = 1.0,
                 reconstruction_weight: float = 1.0,
                 consistency_weight: float = 0.1):
        super().__init__()
        
        self.type_weight = type_weight
        self.strength_weight = strength_weight
        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for human memory network.
        
        Args:
            outputs: Model outputs
            targets: Target values from batch
        
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Memory type classification loss
        if 'memory_type_logits' in outputs and 'memory_type' in targets:
            type_loss = self.ce_loss(outputs['memory_type_logits'], targets['memory_type'])
            losses['memory_type'] = type_loss * self.type_weight
        
        # Memory strength prediction loss
        if 'memory_strength' in outputs and 'memory_strength' in targets:
            strength_loss = self.mse_loss(
                outputs['memory_strength'].squeeze(),
                targets['memory_strength']
            )
            losses['memory_strength'] = strength_loss * self.strength_weight
        
        # Sequence reconstruction loss
        if 'sequence_logits' in outputs and 'sequence' in targets:
            batch_size, seq_len, vocab_size = outputs['sequence_logits'].shape
            recon_loss = self.ce_loss(
                outputs['sequence_logits'].view(-1, vocab_size),
                targets['sequence'].view(-1)
            )
            losses['reconstruction'] = recon_loss * self.reconstruction_weight
        
        # Memory consistency loss (encourages stable memory representations)
        if 'encoded_sequence' in outputs and 'retrieved_memory' in outputs:
            consistency_loss = self.l1_loss(
                outputs['encoded_sequence'],
                outputs['retrieved_memory']
            )
            losses['consistency'] = consistency_loss * self.consistency_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses

class MemoryTrainer:
    """
    Trainer class for Human Memory Network with MacBook Pro optimizations.
    """
    
    def __init__(self,
                 model: HumanMemoryNet,
                 train_loader,
                 val_loader,
                 test_loader,
                 config: Dict):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Setup device (optimized for MacBook Pro)
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup loss function and optimizer
        self.criterion = MemoryLoss(**config.get('loss_weights', {}))
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup logging
        self.experiment_name = f"memory_net_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join('runs', self.experiment_name)
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'memory_stats': []
        }
        
        print(f"Trainer initialized for device: {self.device}")
        print(f"Experiment: {self.experiment_name}")
    
    def _setup_device(self) -> torch.device:
        """Setup device with MacBook Pro optimizations."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with MacBook-friendly settings."""
        optimizer_config = self.config.get('optimizer', {})
        
        if optimizer_config.get('type', 'adamw') == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config.get('type') == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute losses
            losses = self.criterion(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for loss_name, loss_value in losses.items():
                if loss_name not in total_losses:
                    total_losses[loss_name] = 0.0
                total_losses[loss_name] += loss_value.item()
            
            # Log batch statistics
            if batch_idx % self.config.get('log_interval', 50) == 0:
                self._log_batch_stats(batch_idx, num_batches, losses)
        
        # Average losses over epoch
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_losses = {}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute losses
                losses = self.criterion(outputs, batch)
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in total_losses:
                        total_losses[loss_name] = 0.0
                    total_losses[loss_name] += loss_value.item()
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def test(self) -> Dict[str, float]:
        """Test the model and compute detailed metrics."""
        self.model.eval()
        test_metrics = {
            'total_loss': 0.0,
            'type_accuracy': 0.0,
            'strength_mae': 0.0,
            'reconstruction_accuracy': 0.0
        }
        
        num_batches = len(self.test_loader)
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch)
                losses = self.criterion(outputs, batch)
                
                batch_size = batch['sequence'].size(0)
                total_samples += batch_size
                
                # Accumulate losses
                test_metrics['total_loss'] += losses['total'].item()
                
                # Memory type accuracy
                if 'memory_type_logits' in outputs:
                    predicted_types = outputs['memory_type_logits'].argmax(dim=1)
                    correct_types = (predicted_types == batch['memory_type']).sum().item()
                    test_metrics['type_accuracy'] += correct_types
                
                # Memory strength MAE
                if 'memory_strength' in outputs:
                    strength_mae = torch.abs(
                        outputs['memory_strength'].squeeze() - batch['memory_strength']
                    ).sum().item()
                    test_metrics['strength_mae'] += strength_mae
                
                # Sequence reconstruction accuracy
                if 'sequence_logits' in outputs:
                    predicted_tokens = outputs['sequence_logits'].argmax(dim=-1)
                    correct_tokens = (predicted_tokens == batch['sequence']).float().mean().item()
                    test_metrics['reconstruction_accuracy'] += correct_tokens * batch_size
        
        # Average metrics
        test_metrics['total_loss'] /= num_batches
        test_metrics['type_accuracy'] /= total_samples
        test_metrics['strength_mae'] /= total_samples
        test_metrics['reconstruction_accuracy'] /= total_samples
        
        return test_metrics
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch results
            epoch_time = time.time() - start_time
            self._log_epoch_stats(train_losses, val_losses, epoch_time)
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self._save_checkpoint('best_model.pth')
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Final evaluation
        print("\nRunning final evaluation...")
        test_metrics = self.test()
        self._log_test_results(test_metrics)
        
        print(f"Training completed! Best validation loss: {self.best_val_loss:.4f}")
    
    def _log_batch_stats(self, batch_idx: int, num_batches: int, losses: Dict):
        """Log batch-level statistics."""
        progress = 100.0 * batch_idx / num_batches
        print(f"Epoch {self.epoch}, Batch {batch_idx}/{num_batches} ({progress:.1f}%) - "
              f"Loss: {losses['total'].item():.4f}")
    
    def _log_epoch_stats(self, train_losses: Dict, val_losses: Dict, epoch_time: float):
        """Log epoch-level statistics."""
        # Console output
        print(f"\nEpoch {self.epoch + 1} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val Loss: {val_losses['total']:.4f}")
        
        # TensorBoard logging
        for loss_name, loss_value in train_losses.items():
            self.writer.add_scalar(f'Train/{loss_name}', loss_value, self.epoch)
        
        for loss_name, loss_value in val_losses.items():
            self.writer.add_scalar(f'Validation/{loss_name}', loss_value, self.epoch)
        
        # Log memory statistics
        memory_stats = self.model.memory_module.get_memory_stats()
        for stat_name, stat_value in memory_stats.items():
            self.writer.add_scalar(f'Memory/{stat_name}', stat_value, self.epoch)
        
        # Update training history
        self.training_history['train_loss'].append(train_losses['total'])
        self.training_history['val_loss'].append(val_losses['total'])
        self.training_history['memory_stats'].append(memory_stats)
    
    def _log_test_results(self, test_metrics: Dict):
        """Log final test results."""
        print("\nFinal Test Results:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
            self.writer.add_scalar(f'Test/{metric_name}', metric_value, self.epoch)
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, os.path.join('checkpoints', filename))
        print(f"Checkpoint saved: {filename}")
    
    def plot_training_curves(self):
        """Plot training curves."""
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Train Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Memory utilization
        plt.subplot(1, 2, 2)
        memory_utilization = [stats['memory_utilization'] for stats in self.training_history['memory_stats']]
        plt.plot(memory_utilization, label='Memory Utilization')
        plt.xlabel('Epoch')
        plt.ylabel('Utilization')
        plt.title('Memory Module Utilization')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.show()

def main():
    """Main training function optimized for MacBook Pro 2018."""
    
    # Configuration optimized for MacBook Pro 2018 (16GB RAM)
    config = {
        # Model parameters
        'vocab_size': 1000,
        'embed_dim': 128,
        'hidden_dim': 256,
        'memory_size': 1000,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.1,
        'max_sequence_length': 100,
        
        # Training parameters
        'epochs': 50,
        'batch_size': 16,  # Reduced for MacBook
        'num_workers': 2,
        'log_interval': 25,
        'save_interval': 10,
        
        # Optimizer parameters
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999)
        },
        
        # Scheduler parameters
        'scheduler': {
            'type': 'cosine',
            'min_lr': 1e-6
        },
        
        # Loss weights
        'loss_weights': {
            'type_weight': 1.0,
            'strength_weight': 1.0,
            'reconstruction_weight': 1.0,
            'consistency_weight': 0.1
        },
        
        # Dataset parameters
        'train_size': 3000,  # Reduced for MacBook
        'val_size': 600,
        'test_size': 200
    }
    
    print("Initializing Human Memory Network Training")
    print("=" * 50)
    
    # Create data loaders
    print("Creating datasets...")
    train_loader, val_loader, test_loader = create_memory_dataloaders(
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create model
    print("Creating model...")
    model = create_memory_model(config)
    
    # Create trainer
    trainer = MemoryTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    # Start training
    trainer.train(config['epochs'])
    
    # Plot results
    trainer.plot_training_curves()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
