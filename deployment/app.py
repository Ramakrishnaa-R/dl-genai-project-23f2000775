import re
import string

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OPTION_LABELS = ["A", "B", "C", "D", "E"]


def clean_text(text):
    text = text.lower()

    for mark in string.punctuation:
        text = text.replace(mark, " ")

    text = re.sub(r"\s+", " ", text).strip()

    return text


@st.cache_resource
def load_solver():
    return {
        "option_labels": OPTION_LABELS,
        "vectorizer_params": {"stop_words": "english"},
    }


def predict_options(question, options, solver):
    question_clean = clean_text(question)
    options_clean = [clean_text(option) for option in options]

    texts = [question_clean] + options_clean
    vectorizer = TfidfVectorizer(**solver["vectorizer_params"])

    try:
        vectors = vectorizer.fit_transform(texts)
        question_vector = vectors[0]
        option_vectors = vectors[1:]
        scores = cosine_similarity(question_vector, option_vectors)[0]
    except ValueError:
        scores = np.zeros(len(options_clean))

    order = np.argsort(scores)[::-1]
    ranked_predictions = [
        {
            "rank": rank + 1,
            "option": solver["option_labels"][index],
            "answer": options[index],
            "score": float(scores[index]),
        }
        for rank, index in enumerate(order)
    ]

    return ranked_predictions


st.set_page_config(page_title="Smart MCQ Solver", page_icon="?", layout="centered")

st.title("Smart MCQ Solver")

solver = load_solver()

question = st.text_area("Question", height=120, placeholder="Enter the question here")

options = []
for label in OPTION_LABELS:
    option = st.text_input(f"Option {label}", placeholder=f"Enter option {label}")
    options.append(option)

if st.button("Predict", type="primary"):
    missing_fields = []

    if not question.strip():
        missing_fields.append("Question")

    for label, option in zip(OPTION_LABELS, options):
        if not option.strip():
            missing_fields.append(f"Option {label}")

    if missing_fields:
        st.warning("Please fill in: " + ", ".join(missing_fields))
    else:
        ranked_predictions = predict_options(question, options, solver)
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
            st.write(
                f"Option {prediction['option']}: {prediction['score']:.4f}"
            )
