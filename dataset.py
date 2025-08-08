import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict
import json

class MemoryDataset(Dataset):
    """
    Enhanced Memory Dataset that simulates both episodic and semantic memory patterns.
    Optimized for MacBook Pro 2018 with 16GB RAM - uses efficient memory management.
    """
    
    def __init__(self, 
                 num_samples: int = 5000,
                 sequence_length: int = 10,
                 vocab_size: int = 1000,
                 embedding_dim: int = 128,
                 memory_types: List[str] = ['episodic', 'semantic'],
                 seed: int = 42):
        """
        Initialize the memory dataset with configurable parameters.
        
        Args:
            num_samples: Total number of memory sequences (reduced for MacBook)
            sequence_length: Length of each memory sequence
            vocab_size: Size of vocabulary for token generation
            embedding_dim: Dimension of embeddings
            memory_types: Types of memory to simulate
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.memory_types = memory_types
        
        # Generate curated datasets
        self.data = self._generate_memory_data()
        
        # Create embedding layer for efficient token representation
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        print(f"Generated {len(self.data)} memory samples")
        print(f"Memory footprint: ~{self._estimate_memory_mb():.1f} MB")
    
    def _generate_memory_data(self) -> List[Dict]:
        """Generate curated memory data simulating human memory patterns."""
        data = []
        
        # Semantic Memory: Factual knowledge and concepts
        semantic_patterns = self._create_semantic_patterns()
        
        # Episodic Memory: Personal experiences and events
        episodic_patterns = self._create_episodic_patterns()
        
        for i in range(self.num_samples):
            if i % 2 == 0:  # Alternate between memory types
                memory_type = 'semantic'
                pattern = random.choice(semantic_patterns)
            else:
                memory_type = 'episodic'
                pattern = random.choice(episodic_patterns)
            
            # Create sequence with some noise for realism
            sequence = self._create_sequence_from_pattern(pattern)
            
            # Add temporal context (important for human memory)
            temporal_context = self._generate_temporal_context()
            
            # Add emotional valence (affects memory strength)
            emotional_valence = random.uniform(-1, 1)
            
            data.append({
                'sequence': sequence,
                'memory_type': memory_type,
                'pattern': pattern,
                'temporal_context': temporal_context,
                'emotional_valence': emotional_valence,
                'memory_strength': abs(emotional_valence) + random.uniform(0.1, 0.9)
            })
        
        return data
    
    def _create_semantic_patterns(self) -> List[List[int]]:
        """Create patterns representing semantic knowledge."""
        patterns = []
        
        # Mathematical concepts
        math_concepts = [
            [1, 2, 3, 4, 5],  # Numbers sequence
            [10, 20, 30, 40, 50],  # Multiples
            [2, 4, 8, 16, 32],  # Powers of 2
        ]
        
        # Language concepts
        language_concepts = [
            [100, 101, 102, 103, 104],  # Article-noun-verb patterns
            [200, 201, 202, 203, 204],  # Subject-predicate patterns
        ]
        
        # Scientific concepts
        science_concepts = [
            [300, 301, 302, 303, 304],  # Classification hierarchies
            [400, 401, 402, 403, 404],  # Cause-effect relationships
        ]
        
        patterns.extend(math_concepts)
        patterns.extend(language_concepts)
        patterns.extend(science_concepts)
        
        return patterns
    
    def _create_episodic_patterns(self) -> List[List[int]]:
        """Create patterns representing episodic memories."""
        patterns = []
        
        # Daily routine patterns
        routine_patterns = [
            [500, 501, 502, 503, 504],  # Morning routine
            [510, 511, 512, 513, 514],  # Work routine
            [520, 521, 522, 523, 524],  # Evening routine
        ]
        
        # Event sequence patterns
        event_patterns = [
            [600, 601, 602, 603, 604],  # Social events
            [610, 611, 612, 613, 614],  # Learning experiences
            [620, 621, 622, 623, 624],  # Travel experiences
        ]
        
        # Emotional memory patterns
        emotional_patterns = [
            [700, 701, 702, 703, 704],  # Positive experiences
            [710, 711, 712, 713, 714],  # Challenging experiences
        ]
        
        patterns.extend(routine_patterns)
        patterns.extend(event_patterns)
        patterns.extend(emotional_patterns)
        
        return patterns
    
    def _create_sequence_from_pattern(self, pattern: List[int]) -> torch.Tensor:
        """Convert pattern to sequence with noise and padding."""
        # Add some noise to make it realistic
        noisy_pattern = pattern.copy()
        
        # Randomly modify some elements (memory degradation)
        for i in range(len(noisy_pattern)):
            if random.random() < 0.1:  # 10% chance of modification
                noisy_pattern[i] = random.randint(0, self.vocab_size - 1)
        
        # Pad or truncate to sequence_length
        if len(noisy_pattern) < self.sequence_length:
            # Pad with zeros
            noisy_pattern.extend([0] * (self.sequence_length - len(noisy_pattern)))
        else:
            # Truncate
            noisy_pattern = noisy_pattern[:self.sequence_length]
        
        return torch.tensor(noisy_pattern, dtype=torch.long)
    
    def _generate_temporal_context(self) -> Dict:
        """Generate temporal context for memory (time of day, recency, etc.)."""
        return {
            'time_of_day': random.randint(0, 23),  # Hour of day
            'day_of_week': random.randint(0, 6),   # Day of week
            'recency': random.uniform(0, 1),       # How recent (1 = very recent)
            'frequency': random.uniform(0, 1)      # How often accessed
        }
    
    def _estimate_memory_mb(self) -> float:
        """Estimate memory footprint in MB."""
        # Rough estimation for monitoring
        bytes_per_sample = (
            self.sequence_length * 8 +  # sequence tensor
            64 +  # temporal context dict
            32    # other metadata
        )
        total_bytes = bytes_per_sample * len(self.data)
        return total_bytes / (1024 * 1024)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a memory sample."""
        sample = self.data[idx]
        
        # Convert to tensors for training
        result = {
            'sequence': sample['sequence'],
            'memory_type': torch.tensor(1 if sample['memory_type'] == 'episodic' else 0),
            'temporal_features': torch.tensor([
                sample['temporal_context']['time_of_day'] / 23.0,
                sample['temporal_context']['day_of_week'] / 6.0,
                sample['temporal_context']['recency'],
                sample['temporal_context']['frequency']
            ], dtype=torch.float32),
            'emotional_valence': torch.tensor(sample['emotional_valence'], dtype=torch.float32),
            'memory_strength': torch.tensor(sample['memory_strength'], dtype=torch.float32)
        }
        
        return result
    
    def get_memory_statistics(self) -> Dict:
        """Get statistics about the memory dataset."""
        episodic_count = sum(1 for item in self.data if item['memory_type'] == 'episodic')
        semantic_count = len(self.data) - episodic_count
        
        emotional_values = [item['emotional_valence'] for item in self.data]
        strength_values = [item['memory_strength'] for item in self.data]
        
        return {
            'total_samples': len(self.data),
            'episodic_memories': episodic_count,
            'semantic_memories': semantic_count,
            'avg_emotional_valence': np.mean(emotional_values),
            'avg_memory_strength': np.mean(strength_values),
            'memory_footprint_mb': self._estimate_memory_mb()
        }

def create_memory_dataloaders(
    train_size: int = 4000,
    val_size: int = 800,
    test_size: int = 200,
    batch_size: int = 32,
    num_workers: int = 2  # Reduced for MacBook
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders optimized for MacBook Pro.
    
    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        batch_size: Batch size for training
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = MemoryDataset(num_samples=train_size, seed=42)
    val_dataset = MemoryDataset(num_samples=val_size, seed=43)
    test_dataset = MemoryDataset(num_samples=test_size, seed=44)
    
    # Create dataloaders with MacBook-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print("Dataloaders created successfully!")
    print(f"Train: {len(train_loader)} batches")
    print(f"Validation: {len(val_loader)} batches") 
    print(f"Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset
    dataset = MemoryDataset(num_samples=100)
    stats = dataset.get_memory_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test a sample
    sample = dataset[0]
    print(f"\nSample structure:")
    for key, value in sample.items():
        print(f"  {key}: {value}")
