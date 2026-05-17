# NikqTranslate

A simple Neural Machine Translation (NMT) project built from scratch using PyTorch.

This project was created for learning purposes to better understand how machine translation models work internally, including:

* Tokenization
* Vocabulary building
* Encoder–Decoder architecture
* LSTM
* Seq2Seq
* Attention mechanism
* Teacher forcing
* Training loop
* Inference

The current implementation translates English → Vietnamese.

---

# Features

* Custom tokenizer implementation
* Custom vocabulary building
* English and Vietnamese tokenizers
* Encoder–Decoder Seq2Seq architecture
* LSTM-based translation model
* Attention mechanism
* Training loop from scratch
* Inference pipeline
* Hugging Face authentication support
* uv + pyproject.toml workflow

---

# Project Structure

```text
NikqTranslate/
│
├── data/
│   └── corpus.json
│
├── models/
│   └── seq2seq.pt
│
├── src/
│   ├── attention.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── utils.py
│   └── __init__.py
│
├── test/
│   ├── test_lstm.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── __init__.py
│
├── app.py
├── inference.py
├── train.py
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Installation

This project uses:

* Python
* PyTorch
* uv
* pyproject.toml

## Clone the repository

```bash
git clone <repository-url>
cd NikqTranslate
```

## Create virtual environment

```bash
uv venv
```

## Activate environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install dependencies

```bash
uv sync
```

---

# Dataset Format

The dataset is stored in JSON format.

Example:

```json
[
  {
    "en": "hello",
    "vi": "xin chào"
  },
  {
    "en": "i love chemistry",
    "vi": "tôi yêu hóa_học"
  }
]
```

---

# Tokenizer

The project includes a custom tokenizer implementation.

Current tokenizer behavior:

* lowercases text
* splits words by whitespace
* builds vocabulary manually
* supports special tokens

Special tokens:

| Token   | Purpose           |
| ------- | ----------------- |
| `<pad>` | padding           |
| `<sos>` | start of sentence |
| `<eos>` | end of sentence   |
| `<unk>` | unknown token     |

---

# Architecture

The current model architecture:

```text
English sentence
↓
Tokenizer
↓
Encoder (LSTM)
↓
Attention
↓
Decoder (LSTM)
↓
Vietnamese sentence
```

---

# Encoder

The encoder:

* embeds source tokens
* processes them using an LSTM
* returns:

  * encoder outputs
  * hidden state
  * cell state

---

# Attention Mechanism

The attention module allows the decoder to:

* focus on relevant encoder outputs
* avoid compressing the entire sentence into a single vector
* improve long-sequence translation

The decoder computes attention weights over encoder outputs at every timestep.

---

# Decoder

The decoder:

* receives one token at a time
* uses attention context
* predicts the next token
* performs autoregressive generation

---

# Teacher Forcing

During training, the decoder uses:

```text
Ground truth target tokens
```

instead of:

```text
Its own predictions
```

This stabilizes training and improves convergence.

---

# Training

Training uses:

* CrossEntropyLoss
* Adam optimizer
* teacher forcing
* backpropagation

## Start training

```bash
uv run main.py train
```

---

# Inference

Inference performs autoregressive translation.

The model:

1. encodes the source sentence
2. starts with `<sos>`
3. predicts tokens one-by-one
4. stops when `<eos>` is generated

## Run inference

```bash
uv run main.py inference
```

---

# Testing

## Test tokenizer encode functionality

```bash
uv run main.py test-tokenizer-encode
```

# Test tokenizer decode functionality

```bash
uv run main.py test-tokenizer-decode
```

## Test encoder

```bash
uv run main.py test-encoder
```

## Test decoder

```bash
uv run main.py test-decoder
```

## Test Seq2Seq model

```bash
uv run main.py test-seq2seq
```

---

# Model Saving

The trained model is saved using:

```python
torch.save(model.state_dict(), "models/seq2seq.pt")
```

Load model:

```python
model.load_state_dict(torch.load("models/seq2seq.pt"))
```

---

# Current Limitations

Current limitations include:

* very small dataset
* word-level tokenizer
* limited generalization
* no beam search
* no BLEU evaluation
* no Transformer architecture
* no subword tokenizer

Because of the small dataset, the model often memorizes training examples instead of fully generalizing.

---

# Future Improvements

Planned improvements:

* larger dataset
* SentencePiece / BPE tokenizer
* bidirectional encoder
* beam search decoding
* BLEU score evaluation
* Transformer architecture
* attention visualization
* better Vietnamese tokenization
* Hugging Face dataset integration

---

# Technologies Used

* Python
* PyTorch
* uv
* Hugging Face
* dotenv

---

# Learning Goals

This project focuses on understanding the internal mechanics of:

* neural machine translation
* sequence modeling
* encoder–decoder architectures
* attention mechanisms
* autoregressive generation

instead of achieving production-level translation quality.

---

# Author

NikqTranslate was built as a personal NLP learning project.
