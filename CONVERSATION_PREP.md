# Conversation Preparation: Data Format & Implementation Overview

## Quick Reference (Key Numbers)

- **Current Dataset**: 18,039 human protein sequences (151,602 lines total)
- **Vocabulary Size**: 23 tokens (20 amino acids + 3 special tokens)
- **Max Sequence Length**: 1024 tokens
- **Masking Strategy**: 15% random token masking
- **S4 Variants**: Diagonal (S4D) and NPLR implementations ready
- **Status**: Dataloader complete, S4 model architecture pending

---

## Data Format (FASTA File)

### What You Have
- **File**: `data/protein/train.fasta`
- **Format**: Standard FASTA format
- **Current Content**: Human proteins from UniProt

### FASTA Format Structure
```
>sp|A0A087X1C5|CP2D7_HUMAN Cytochrome P450 2D7 OS=Homo sapiens OX=9606 GN=CYP2D7 PE=1 SV=1
MGLEALVPLAMIVAIFLLLVDLMHRHQRWAARYPPGPLPLPGLGNLLHVDFQNTPYCFDQ
LRRRFGDVFSLQLAWTPVVVLNGLAAVREAMVTRGEDTADRPPAPIYQVLGFGPRSQGVI
...
```

**Key Points:**
- Lines starting with `>` are headers (metadata: protein ID, name, organism, gene name)
- Lines after headers are amino acid sequences (single-letter codes)
- Sequences can span multiple lines
- Each entry = one protein sequence

### What's in Your Data
- **Format**: Amino acid sequences in single-letter code (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
- **Current Filter**: Human proteins only (Homo sapiens)
- **Size**: 151,602 total lines, **18,039 protein sequences**
- **Sequence Lengths**: Variable (your dataloader handles up to 1024 amino acids)

---

## What You've Implemented

### 1. **FASTA Parser** (`parse_fasta`)
- Reads FASTA files line by line
- Extracts headers and sequences
- Returns list of (header, sequence) tuples
- Handles multi-line sequences

### 2. **Protein Tokenizer** (`ProteinTokenizer`)
- **Vocabulary**: 23 tokens total
  - 20 standard amino acids (A-Y)
  - 3 special tokens: `<PAD>` (0), `<MASK>` (1), `<UNK>` (2)
- **Functions**:
  - `encode()`: Converts amino acid string → integer IDs
  - `decode()`: Converts integer IDs → amino acid string
- Maps each amino acid to unique integer (3-22)

### 3. **Protein Dataset** (`ProteinDataset`)
- PyTorch Dataset class
- **Processing Pipeline**:
  1. Loads FASTA file
  2. Tokenizes sequences
  3. Filters sequences ≤ 1024 tokens (configurable `l_max`)
  4. Supports caching processed sequences
- **Autocompletion Task**:
  - Randomly masks 15% of tokens (configurable `mask_prob`)
  - Returns: `(input_ids, target_ids, attention_mask)`
  - Input has masked positions, target has original sequence
- **Padding/Truncation**: All sequences padded/truncated to `l_max=1024`

### 4. **DataLoader Helper** (`create_dataloader`)
- Wraps dataset in PyTorch DataLoader
- Configurable: batch size, workers, shuffling
- Ready for training loop

---

## S4 Model Implementation (What You Already Have)

### Core Components

1. **SSKernelDiag** (S4D - Diagonal variant)
   - Efficient diagonal state space kernel
   - Uses Vandermonde matrix for convolution
   - Supports both convolution and recurrent modes

2. **SSKernelNPLR** (Original S4)
   - Normal Plus Low-Rank parameterization
   - More expressive but computationally heavier
   - Uses Cauchy kernel trick

3. **Discretization Methods**
   - `discretize_zoh`: Zero-order hold (more accurate)
   - `discretize_bilinear`: Bilinear transform

4. **HiPPO Initialization**
   - HiPPO-LegS matrices for optimal long-range dependencies
   - Initializes state space parameters

### Model Architecture (Not Yet Implemented)
- **Missing**: Full S4 model wrapper that:
  - Takes tokenized sequences as input
  - Embeds tokens to `d_model` dimensions
  - Applies S4 layers
  - Outputs predictions for masked tokens

---

## How Everything Fits Together

```
FASTA File → Parser → Tokenizer → Dataset → DataLoader
                                              ↓
                                    [S4 Model] ← (to be implemented)
                                              ↓
                                    Predictions
```

**Current Flow:**
1. FASTA file contains raw protein sequences
2. Parser extracts sequences
3. Tokenizer converts amino acids → integers
4. Dataset applies masking and padding
5. DataLoader batches sequences
6. **Next Step**: S4 model processes batches and predicts masked tokens

---

## Talking Points for Conversation

### About Your Data
- "I'm using standard FASTA format from UniProt"
- "Currently have human proteins, but understand we need more diversity"
- "The dataloader handles variable-length sequences by padding/truncating to 1024 tokens"
- "Each sequence is tokenized into integers (20 amino acids + 3 special tokens)"

### About Your Implementation
- "I've implemented a complete dataloader pipeline: FASTA parsing, tokenization, and masking for autocompletion"
- "Using masked language modeling approach - randomly mask 15% of tokens and predict them"
- "The S4 kernel implementation is ready - supports both diagonal (S4D) and NPLR variants"
- "Next step is building the full model architecture that connects embeddings → S4 layers → predictions"

### Questions to Ask
- "What diversity should I aim for in the training set? (bacterial, eukaryotic, insect proportions?)"
- "Should I filter by sequence length or include all sequences?"
- "Any specific protein families or domains to prioritize?"
- "What masking strategy works best for protein autocompletion? (current: 15% random masking)"

### Technical Details You Can Mention
- "Vocabulary size: 23 tokens (20 amino acids + 3 special tokens)"
- "Max sequence length: 1024 tokens (configurable)"
- "Masking probability: 15% (configurable)"
- "S4 supports both convolution and recurrent inference modes"
- "Using HiPPO initialization for long-range dependencies"

---

## Sample Responses

**Q: "What format is your data in?"**
> "Standard FASTA format. Each entry has a header line starting with '>' containing metadata, followed by the amino acid sequence in single-letter code. My parser extracts both headers and sequences, then tokenizes the sequences into integers for the model."

**Q: "How are you handling the sequences?"**
> "I tokenize each amino acid to an integer ID. The vocabulary has 23 tokens - 20 for standard amino acids plus special tokens for padding, masking, and unknown characters. Sequences are padded or truncated to 1024 tokens, and I apply random masking for the autocompletion task."

**Q: "What's your masking strategy?"**
> "I'm using a masked language modeling approach similar to BERT. For each sequence, I randomly mask 15% of the tokens, and the model will learn to predict those masked positions. This teaches the model to understand protein sequence context and dependencies."

**Q: "What do you still need to implement?"**
> "The S4 kernel computation is done - I have both the diagonal and NPLR variants. What's missing is the full model architecture: embedding layer to convert token IDs to vectors, S4 layers to process sequences, and output head to predict masked tokens. The dataloader is ready to feed data into this model once it's built."

