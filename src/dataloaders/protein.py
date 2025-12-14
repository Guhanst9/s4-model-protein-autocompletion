import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import random


class ProteinTokenizer:
    """amino acid tokenizer for protein sequences"""
    
    def __init__(self):
        # standard 20 amino acids
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        # special tokens
        self.pad_token = '<PAD>'
        self.mask_token = '<MASK>'
        self.unk_token = '<UNK>'
        
        # build vocab
        self.vocab = {self.pad_token: 0, self.mask_token: 1, self.unk_token: 2}
        for i, aa in enumerate(self.amino_acids):
            self.vocab[aa] = i + 3
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, sequence: str) -> List[int]:
        """encode amino acid sequence to token ids"""
        tokens = []
        for aa in sequence.upper():
            if aa in self.vocab:
                tokens.append(self.vocab[aa])
            else:
                tokens.append(self.vocab[self.unk_token])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """decode token ids to amino acid sequence"""
        return ''.join([self.idx_to_token[idx] for idx in token_ids if idx != self.vocab[self.pad_token]])


def parse_fasta(fasta_file: str) -> List[Tuple[str, str]]:
    """parse fasta file and return list of (header, sequence) tuples"""
    sequences = []
    current_header = None
    current_sequence = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_sequence)))
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        
        if current_header is not None:
            sequences.append((current_header, ''.join(current_sequence)))
    
    return sequences


class ProteinDataset(Dataset):
    """dataset for protein sequence autocompletion"""
    
    def __init__(
        self,
        fasta_file: str,
        tokenizer: Optional[ProteinTokenizer] = None,
        l_max: int = 1024,
        mask_prob: float = 0.15,
        cache_dir: Optional[str] = None,
    ):
        self.fasta_file = fasta_file
        self.l_max = l_max
        self.mask_prob = mask_prob
        
        if tokenizer is None:
            self.tokenizer = ProteinTokenizer()
        else:
            self.tokenizer = tokenizer
        
        # load sequences
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, 'sequences.pt')
        
        if cache_file and os.path.exists(cache_file):
            self.sequences = torch.load(cache_file)
        else:
            fasta_data = parse_fasta(fasta_file)
            self.sequences = []
            for header, seq in fasta_data:
                if len(seq) > 0:
                    encoded = self.tokenizer.encode(seq)
                    if len(encoded) <= l_max:
                        self.sequences.append(encoded)
            
            if cache_file:
                torch.save(self.sequences, cache_file)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return (input_ids, target_ids, attention_mask)"""
        sequence = self.sequences[idx].copy()
        seq_len = len(sequence)
        
        # pad or truncate to l_max
        if seq_len < self.l_max:
            padded = sequence + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (self.l_max - seq_len)
        else:
            padded = sequence[:self.l_max]
        
        # create input and target
        input_ids = padded.copy()
        target_ids = padded.copy()
        attention_mask = [1] * seq_len + [0] * (self.l_max - seq_len)
        
        # apply masking for autocompletion
        num_mask = max(1, int(seq_len * self.mask_prob))
        mask_positions = random.sample(range(seq_len), min(num_mask, seq_len))
        
        for pos in mask_positions:
            input_ids[pos] = self.tokenizer.vocab[self.tokenizer.mask_token]
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )


def create_dataloader(
    fasta_file: str,
    tokenizer: Optional[ProteinTokenizer] = None,
    l_max: int = 1024,
    mask_prob: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """create dataloader for protein sequences"""
    dataset = ProteinDataset(
        fasta_file=fasta_file,
        tokenizer=tokenizer,
        l_max=l_max,
        mask_prob=mask_prob,
        cache_dir=cache_dir,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

