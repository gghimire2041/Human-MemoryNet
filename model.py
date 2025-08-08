import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

class MemoryAttention(nn.Module):
    """
    Multi-head attention mechanism for memory retrieval.
    Simulates how humans selectively recall relevant memories.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query: torch.Tensor, memory: torch.Tensor, 
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, embed_dim] - what we're trying to remember
            memory: [batch_size, memory_size, embed_dim] - stored memories
            memory_mask: [batch_size, memory_size] - mask for valid memories
        
        Returns:
            retrieved_memory: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, num_heads, seq_len, memory_size]
        """
        batch_size, seq_len, _ = query.shape
        memory_size = memory.shape[1]
        
        # Project to query, key, value
        Q = self.query_projection(query)
        K = self.key_projection(memory)
        V = self.value_projection(memory)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, memory_size, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, memory_size, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply memory mask if provided
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, memory_size]
            attention_scores.masked_fill_(memory_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = self.output_projection(attended_values)
        
        return output, attention_weights

class ExternalMemoryModule(nn.Module):
    """
    External memory module that simulates human long-term memory storage.
    Includes mechanisms for memory consolidation and forgetting.
    """
    
    def __init__(self, 
                 memory_size: int = 1000,  # Reduced for MacBook Pro
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 consolidation_rate: float = 0.01):
        super().__init__()
        
        self.memory_size = memory_size
        self.embed_dim = embed_dim
        self.consolidation_rate = consolidation_rate
        
        # Initialize memory matrix
        self.memory_matrix = nn.Parameter(
            torch.randn(memory_size, embed_dim) * 0.1
        )
        
        # Memory usage tracking (simulates forgetting)
        self.register_buffer('memory_usage', torch.zeros(memory_size))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        
        # Attention mechanism for memory retrieval
        self.attention = MemoryAttention(embed_dim, num_heads)
        
        # Memory gating (controls what gets stored)
        self.memory_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Memory addressing (where to store new memories)
        self.address_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, memory_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                input_sequence: torch.Tensor,
                store_memories: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_sequence: [batch_size, seq_len, embed_dim]
            store_memories: Whether to store new memories
        
        Returns:
            Dictionary containing retrieved memories and attention weights
        """
        batch_size, seq_len, embed_dim = input_sequence.shape
        
        # Create memory mask (all memories are valid for now)
        memory_mask = torch.ones(batch_size, self.memory_size, device=input_sequence.device)
        
        # Expand memory matrix for batch processing
        expanded_memory = self.memory_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Retrieve relevant memories using attention
        retrieved_memory, attention_weights = self.attention(
            input_sequence, expanded_memory, memory_mask
        )
        
        # Store new memories if requested
        if store_memories and self.training:
            self._store_memories(input_sequence)
        
        # Update memory usage and age
        self._update_memory_statistics()
        
        return {
            'retrieved_memory': retrieved_memory,
            'attention_weights': attention_weights,
            'memory_matrix': self.memory_matrix,
            'memory_usage': self.memory_usage
        }
    
    def _store_memories(self, input_sequence: torch.Tensor):
        """Store new memories in the external memory matrix."""
        # Average across sequence and batch for storage
        memory_to_store = input_sequence.mean(dim=(0, 1))  # [embed_dim]
        
        # Decide whether to store (memory gating)
        gate_value = self.memory_gate(memory_to_store)
        
        if gate_value.item() > 0.5:  # Store memory
            # Determine where to store
            address_weights = self.address_network(memory_to_store)
            
            # Update memory matrix using consolidation
            update = memory_to_store.unsqueeze(0) * address_weights.unsqueeze(1)
            self.memory_matrix.data = (
                (1 - self.consolidation_rate) * self.memory_matrix.data +
                self.consolidation_rate * update
            )
            
            # Update usage statistics
            self.memory_usage += address_weights
    
    def _update_memory_statistics(self):
        """Update memory usage and age statistics."""
        # Age all memories
        self.memory_age += 1
        
        # Implement forgetting curve (memories decay over time)
        forgetting_factor = 1 - (self.memory_age / 10000.0).clamp(0, 0.1)
        self.memory_usage *= forgetting_factor
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get statistics about memory usage."""
        return {
            'avg_memory_usage': self.memory_usage.mean().item(),
            'max_memory_usage': self.memory_usage.max().item(),
            'avg_memory_age': self.memory_age.mean().item(),
            'memory_utilization': (self.memory_usage > 0.1).float().mean().item()
        }

class HumanMemoryNet(nn.Module):
    """
    Enhanced Human Memory Network that simulates human memory processes:
    - Encoding: Transform input into memory representations
    - Storage: Store memories in external memory module
    - Retrieval: Retrieve relevant memories using attention
    - Consolidation: Strengthen important memories over time
    """
    
    def __init__(self,
                 vocab_size: int = 1000,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 memory_size: int = 1000,
                 num_heads: int = 8,
                 num_layers: int = 3,  # Reduced for MacBook Pro
                 dropout: float = 0.1,
                 max_sequence_length: int = 100):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # Input embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = self._create_positional_encoding()
        
        # Encoding stage: Transform input to memory representation
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # External memory module
        self.memory_module = ExternalMemoryModule(
            memory_size=memory_size,
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        # Context integration (combines input with retrieved memories)
        self.context_integration = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers for different tasks
        self.memory_type_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # episodic vs semantic
        )
        
        self.memory_strength_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.sequence_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Temporal context processor
        self.temporal_processor = nn.Sequential(
            nn.Linear(4, embed_dim // 4),  # 4 temporal features
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create positional encoding for sequence modeling."""
        pe = torch.zeros(self.max_sequence_length, self.embed_dim)
        position = torch.arange(0, self.max_sequence_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() *
                           -(math.log(10000.0) / self.embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_seq_len, embed_dim]
    
    def forward(self, 
                batch: Dict[str, torch.Tensor],
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Human Memory Network.
        
        Args:
            batch: Dictionary containing input tensors
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing model outputs
        """
        sequence = batch['sequence']  # [batch_size, seq_len]
        temporal_features = batch['temporal_features']  # [batch_size, 4]
        
        batch_size, seq_len = sequence.shape
        device = sequence.device
        
        # Move positional encoding to device
        if not hasattr(self, '_pe_device') or self._pe_device != device:
            self.positional_encoding = self.positional_encoding.to(device)
            self._pe_device = device
        
        # Encoding Stage: Transform input to memory representation
        embedded = self.embedding(sequence)  # [batch_size, seq_len, embed_dim]
        
        # Add positional encoding
        if seq_len <= self.max_sequence_length:
            embedded += self.positional_encoding[:, :seq_len, :]
        
        # Process temporal context
        temporal_context = self.temporal_processor(temporal_features)  # [batch_size, embed_dim]
        temporal_context = temporal_context.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine with temporal context
        encoded_input = embedded + temporal_context
        
        # Apply transformer encoder
        encoded_sequence = self.encoder(encoded_input)  # [batch_size, seq_len, embed_dim]
        
        # Storage and Retrieval Stage: Interact with external memory
        memory_output = self.memory_module(encoded_sequence, store_memories=self.training)
        retrieved_memory = memory_output['retrieved_memory']
        
        # Context Integration: Combine input with retrieved memories
        integrated_context, attention_weights = self.context_integration(
            encoded_sequence,  # query
            retrieved_memory,  # key
            retrieved_memory   # value
        )
        
        # Average pooling for sequence-level representations
        sequence_repr = integrated_context.mean(dim=1)  # [batch_size, embed_dim]
        
        # Generate outputs
        memory_type_logits = self.memory_type_classifier(sequence_repr)
        memory_strength = self.memory_strength_predictor(sequence_repr)
        sequence_logits = self.sequence_decoder(integrated_context)
        
        outputs = {
            'memory_type_logits': memory_type_logits,
            'memory_strength': memory_strength,
            'sequence_logits': sequence_logits,
            'encoded_sequence': encoded_sequence,
            'retrieved_memory': retrieved_memory,
            'integrated_context': integrated_context
        }
        
        if return_attention:
            outputs.update({
                'memory_attention': memory_output['attention_weights'],
                'context_attention': attention_weights
            })
        
        return outputs
    
    def get_model_stats(self) -> Dict[str, any]:
        """Get comprehensive model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim
        }
        
        # Add memory module statistics
        memory_stats = self.memory_module.get_memory_stats()
        model_stats.update({f'memory_{k}': v for k, v in memory_stats.items()})
        
        return model_stats
    
    def reset_memory(self):
        """Reset the external memory (useful for evaluation)."""
        self.memory_module.memory_matrix.data.normal_(0, 0.1)
        self.memory_module.memory_usage.zero_()
        self.memory_module.memory_age.zero_()

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters for memory optimization."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
