"""
Centroid Proximity Weighting (CPW):
Weight each response inversely proportional to its Euclidean distance from the
embedding centroid, then sum weights per answer.
"""

import json
import argparse
from collections import defaultdict, Counter

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from evaluation_utils import answers_match


def load_embedder(model_name):
    """Load tokenizer and model for embedding."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_texts(texts, tokenizer, model):
    """Embed a list of texts using mean pooling of the last hidden state."""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(emb)
    return np.array(embeddings)


def cpw_vote(embeddings, parsed_answers):
    """
    Centroid Proximity Weighting: weight each response inversely proportional
    to its Euclidean distance from the centroid of all embeddings.
    """
    n = len(embeddings)
    if n < 2:
        counts = Counter(a for a in parsed_answers if a is not None)
        return counts.most_common(1)[0][0] if counts else None

    centroid = np.mean(embeddings, axis=0)
    weighted_answers = defaultdict(float)

    for i in range(n):
        if parsed_answers[i] is None:
            continue
        distance = euclidean(embeddings[i], centroid)
        weight = 1.0 / (distance + 1e-8)
        weighted_answers[parsed_answers[i]] += weight

    if not weighted_answers:
        return None
    return max(weighted_answers, key=weighted_answers.get)


def majority_vote(parsed_answers):
    """Standard majority vote baseline."""
    counts = Counter(a for a in parsed_answers if a is not None)
    return counts.most_common(1)[0][0] if counts else None


def evaluate(json_path, embedder_name, dataset_name):
    """Run CPW evaluation on a generated results file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    tokenizer, model = load_embedder(embedder_name)

    correct_majority = 0
    correct_cpw = 0
    total = len(data)

    for item in tqdm(data, desc=f"CPW eval ({dataset_name})"):
        responses = item["responses"]
        texts = [r["response"] for r in responses]
        parsed = [r.get("parsed_answer") for r in responses]

        # Majority vote
        mv_answer = majority_vote(parsed)
        if answers_match(mv_answer, item["true_answer"], dataset_name):
            correct_majority += 1

        # CPW
        embeddings = embed_texts(texts, tokenizer, model)
        cpw_answer = cpw_vote(embeddings, parsed)
        if answers_match(cpw_answer, item["true_answer"], dataset_name):
            correct_cpw += 1

    maj_acc = correct_majority / total * 100
    cpw_acc = correct_cpw / total * 100

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name} | Embedder: {embedder_name}")
    print(f"Majority Vote Accuracy: {maj_acc:.2f}%")
    print(f"CPW Accuracy:           {cpw_acc:.2f}%")
    print(f"Improvement:            {cpw_acc - maj_acc:+.2f}%")
    print(f"{'='*50}")

    return {"majority_vote": maj_acc, "cpw": cpw_acc}


# ── Embedder selection by dataset ───────────────────────────────────────────

EMBEDDER_MAP = {
    "aqua": "allenai/scibert_scivocab_uncased",
    "svamp": "allenai/scibert_scivocab_uncased",
    "strategyqa": "roberta-base",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Centroid Proximity Weighting evaluation")
    parser.add_argument("--json_path", required=True, help="Path to generated responses JSON")
    parser.add_argument("--dataset", required=True, choices=["aqua", "svamp", "strategyqa"])
    parser.add_argument("--embedder", default=None,
                        help="Override embedder model (default: SciBERT for math, RoBERTa for commonsense)")
    return parser.parse_args()


def main():
    args = parse_args()
    embedder = args.embedder or EMBEDDER_MAP[args.dataset]
    evaluate(args.json_path, embedder, args.dataset)


if __name__ == "__main__":
    main()
