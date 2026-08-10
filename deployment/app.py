import json
import os
import re
import string
from pathlib import Path

import numpy as np
import streamlit as st


OPTION_LABELS = ["A", "B", "C", "D", "E"]
APP_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = Path(os.getenv("CNN_MODEL_PATH", APP_DIR / "cnn_model.npz"))
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
            f"CNN checkpoint not found: {model_path}. Set CNN_MODEL_PATH to a saved .npz file."
        )

    model_data = np.load(model_path, allow_pickle=True)

    if "vocabulary" in model_data:
        vocabulary = {str(key): int(value) for key, value in model_data["vocabulary"].item().items()}
    else:
        vocabulary = load_vocab(vocab_path)

    return vocabulary, {
        "embedding_weight": model_data["embedding_weight"],
        "convolution_weight": model_data["convolution_weight"],
        "convolution_bias": model_data["convolution_bias"],
        "output_weight": model_data["output_weight"],
        "output_bias": model_data["output_bias"],
    }


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

    input_ids = np.array([option_sequences], dtype=np.int64)
    scores = score_with_cnn(input_ids, model)

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


def score_with_cnn(input_ids, model):
    batch_size, number_of_options, sequence_length = input_ids.shape
    scores = np.zeros((batch_size, number_of_options), dtype=np.float32)

    embedding_weight = model["embedding_weight"]
    convolution_weight = model["convolution_weight"]
    convolution_bias = model["convolution_bias"]
    output_weight = model["output_weight"].reshape(NUM_FILTERS)
    output_bias = float(model["output_bias"].reshape(()))

    for batch_index in range(batch_size):
        for option_index in range(number_of_options):
            token_ids = input_ids[batch_index, option_index]
            token_ids = token_ids[:MAX_LENGTH]

            embedded = embedding_weight[token_ids]
            if embedded.shape[0] < KERNEL_SIZE:
                pad_width = KERNEL_SIZE - embedded.shape[0]
                embedded = np.pad(embedded, ((0, pad_width), (0, 0)), mode="constant")

            window_scores = []
            for start in range(0, embedded.shape[0] - KERNEL_SIZE + 1):
                window = embedded[start : start + KERNEL_SIZE]
                conv = np.tensordot(convolution_weight, window, axes=([1, 2], [1, 0]))
                conv = conv + convolution_bias
                window_scores.append(np.maximum(conv, 0.0))

            features = np.max(np.stack(window_scores, axis=0), axis=0)
            scores[batch_index, option_index] = float(np.dot(output_weight, features) + output_bias)

    logits = scores[0]
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


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