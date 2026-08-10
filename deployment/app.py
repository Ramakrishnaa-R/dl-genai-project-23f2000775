import json
import os
import re
import string
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn


OPTION_LABELS = ["A", "B", "C", "D", "E"]
APP_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = Path(os.getenv("CNN_MODEL_PATH", APP_DIR / "cnn_model.pt"))
CNN_VOCAB_PATH = Path(os.getenv("CNN_VOCAB_PATH", APP_DIR / "cnn_vocab.json"))
MAX_LENGTH = 128
EMBEDDING_DIM = 32
NUM_FILTERS = 32
KERNEL_SIZE = 3


def clean_text(text):
    text = text.lower()

    for mark in string.punctuation:
        text = text.replace(mark, " ")

    text = re.sub(r"\s+", " ", text).strip()

    return text


class SimpleCNN(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, EMBEDDING_DIM, padding_idx=0)
        self.convolution = nn.Conv1d(EMBEDDING_DIM, NUM_FILTERS, kernel_size=KERNEL_SIZE)
        self.output = nn.Linear(NUM_FILTERS, 1)

    def forward(self, input_ids):
        batch_size, number_of_options, sequence_length = input_ids.shape
        input_ids = input_ids.view(batch_size * number_of_options, sequence_length)
        embedded = self.embedding(input_ids).transpose(1, 2)
        features = torch.relu(self.convolution(embedded))
        pooled = torch.max(features, dim=2).values
        scores = self.output(pooled).squeeze(1)
        return scores.view(batch_size, number_of_options)


def build_default_vocab():
    return {"<PAD>": 0, "<UNK>": 1}


def load_vocab(vocab_path):
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_path}. Set CNN_VOCAB_PATH to a JSON vocab file."
        )

    with vocab_path.open("r", encoding="utf-8") as file_handle:
        vocabulary = json.load(file_handle)

    if not isinstance(vocabulary, dict) or "<PAD>" not in vocabulary or "<UNK>" not in vocabulary:
        raise ValueError("Vocabulary file must be a JSON object containing <PAD> and <UNK> tokens.")

    return {str(key): int(value) for key, value in vocabulary.items()}


def load_model(model_path, vocab_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN checkpoint not found: {model_path}. Set CNN_MODEL_PATH to a saved .pt/.pth file."
        )

    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "vocabulary" in checkpoint:
        vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    else:
        vocabulary = load_vocab(vocab_path)

    model = SimpleCNN(len(vocabulary))
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    return vocabulary, model


def encode_text(text, vocabulary):
    token_ids = []

    for word in clean_text(text).split()[:MAX_LENGTH]:
        token_ids.append(vocabulary.get(word, vocabulary["<UNK>"]))

    while len(token_ids) < MAX_LENGTH:
        token_ids.append(vocabulary["<PAD>"])

    return token_ids


def predict_options(question, options, vocabulary, model):
    option_sequences = []

    for option in options:
        text = f"{question} {option}"
        option_sequences.append(encode_text(text, vocabulary))

    input_ids = torch.tensor([option_sequences], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)
        scores = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    ranked_indices = np.argsort(-scores)
    ranked_predictions = [
        {
            "rank": rank + 1,
            "option": OPTION_LABELS[index],
            "answer": options[index],
            "score": float(scores[index]),
        }
        for rank, index in enumerate(ranked_indices)
    ]

    return ranked_predictions


st.set_page_config(page_title="Smart MCQ Solver", page_icon="?", layout="centered")
st.title("Smart MCQ Solver")
st.caption(f"CNN checkpoint: {CNN_MODEL_PATH}")

model_error = None
vocabulary = None
model = None

try:
    vocabulary, model = load_model(CNN_MODEL_PATH, CNN_VOCAB_PATH)
except Exception as exc:
    model_error = str(exc)

question = st.text_area("Question", height=120, placeholder="Enter the question here")

options = []
for label in OPTION_LABELS:
    option = st.text_input(f"Option {label}", placeholder=f"Enter option {label}")
    options.append(option)

if model_error:
    st.error(model_error)
    st.info(
        "To deploy the CNN model quickly, add the trained checkpoint and vocab file next to this app, or set CNN_MODEL_PATH and CNN_VOCAB_PATH to their locations."
    )
elif st.button("Predict", type="primary"):
    missing_fields = []

    if not question.strip():
        missing_fields.append("Question")

    for label, option in zip(OPTION_LABELS, options):
        if not option.strip():
            missing_fields.append(f"Option {label}")

    if missing_fields:
        st.warning("Please fill in: " + ", ".join(missing_fields))
    else:
        ranked_predictions = predict_options(question, options, vocabulary, model)
        top_3 = ranked_predictions[:3]

        st.subheader("Top-3 Predictions")
        for prediction in top_3:
            st.write(
                f"{prediction['rank']}. Option {prediction['option']} "
                f"(score: {prediction['score']:.4f})"
            )
            st.caption(prediction["answer"])

        st.subheader("Scores for All Options")
        for prediction in ranked_predictions:
            st.write(f"Option {prediction['option']}: {prediction['score']:.4f}")