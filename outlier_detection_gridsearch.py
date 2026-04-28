"""
Outlier Detection + Majority Vote:
Remove degenerate CoT responses via KNN, Isolation Forest, or One-Class SVM
in embedding space, then majority-vote over the remaining responses.
"""

import json
import argparse
from collections import Counter

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


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


# ── Outlier detection methods ───────────────────────────────────────────────

def detect_outliers_isolation_forest(embeddings, params):
    """Returns mask: 1 = inlier, -1 = outlier."""
    clf = IsolationForest(random_state=42, **params)
    clf.fit(embeddings)
    return clf.predict(embeddings)


def detect_outliers_knn(embeddings, params):
    """Returns mask: 1 = inlier, -1 = outlier based on KNN distance threshold."""
    if len(embeddings) <= params["n_neighbors"]:
        return np.ones(len(embeddings))  # not enough points, keep all
    knn = NearestNeighbors(
        n_neighbors=params["n_neighbors"],
        metric=params["metric"],
        algorithm=params["algorithm"],
    )
    knn.fit(embeddings)
    distances, _ = knn.kneighbors(embeddings)
    avg_distances = np.mean(distances, axis=1)
    threshold = np.percentile(avg_distances, params["threshold_percentile"])
    return np.where(avg_distances > threshold, -1, 1)


def detect_outliers_svm(embeddings, params):
    """Returns mask: 1 = inlier, -1 = outlier."""
    clf = OneClassSVM(gamma=params["gamma"], nu=params["nu"], kernel=params["kernel"])
    clf.fit(embeddings)
    return clf.predict(embeddings)


# ── Parameter grids ─────────────────────────────────────────────────────────

PARAM_GRIDS = {
    "isolation_forest": {
        "n_estimators": [50, 100, 200, 300],
        "max_samples": ["auto"],
        "contamination": ["auto", 0.1, 0.2, 0.03],
    },
    "knn": {
        "n_neighbors": [3, 5, 7],
        "metric": ["euclidean", "manhattan"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
        "threshold_percentile": [85, 90, 95],
    },
    "svm": {
        "nu": [0.01, 0.05, 0.1],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear", "sigmoid"],
    },
}

DETECTORS = {
    "isolation_forest": detect_outliers_isolation_forest,
    "knn": detect_outliers_knn,
    "svm": detect_outliers_svm,
}


# ── Evaluation ──────────────────────────────────────────────────────────────

def majority_vote(parsed_answers):
    """Standard majority vote over non-None answers."""
    counts = Counter(a for a in parsed_answers if a is not None)
    return counts.most_common(1)[0][0] if counts else None


def evaluate_with_outlier_detection(data, tokenizer, model, method, params, dataset_name):
    """Run outlier detection + majority vote for one param setting."""
    detect_fn = DETECTORS[method]
    correct = 0
    total = len(data)

    for item in data:
        responses = item["responses"]
        true_answer = str(item["true_answer"]).strip()
        texts = [r["response"] for r in responses]
        parsed = [r.get("parsed_answer") for r in responses]

        embeddings = embed_texts(texts, tokenizer, model)

        try:
            labels = detect_fn(embeddings, params)
        except Exception:
            # Fallback to keeping all if detection fails
            labels = np.ones(len(embeddings))

        # Keep only inliers (label == 1)
        filtered_answers = [parsed[i] for i in range(len(parsed)) if labels[i] == 1]

        answer = majority_vote(filtered_answers) if filtered_answers else majority_vote(parsed)
        if answer is not None and str(answer).strip().lower() == true_answer.lower():
            correct += 1

    return correct / total * 100


def run_gridsearch(json_path, embedder_name, dataset_name, method):
    """Run grid search for one outlier detection method."""
    with open(json_path, "r") as f:
        data = json.load(f)

    tokenizer, model = load_embedder(embedder_name)

    # Pre-compute embeddings to avoid redundant work
    print(f"Pre-computing embeddings with {embedder_name}...")
    all_embeddings = []
    for item in tqdm(data, desc="Embedding"):
        texts = [r["response"] for r in item["responses"]]
        embs = embed_texts(texts, tokenizer, model)
        all_embeddings.append(embs)

    # Baseline majority vote
    correct_mv = 0
    for item in data:
        parsed = [r.get("parsed_answer") for r in item["responses"]]
        true_answer = str(item["true_answer"]).strip()
        mv = majority_vote(parsed)
        if mv is not None and str(mv).strip().lower() == true_answer.lower():
            correct_mv += 1
    mv_acc = correct_mv / len(data) * 100

    print(f"\nMajority Vote Baseline: {mv_acc:.2f}%")
    print(f"\nGrid search: {method}")
    print(f"{'='*70}")

    param_grid = ParameterGrid(PARAM_GRIDS[method])
    best_acc = 0
    best_params = None

    detect_fn = DETECTORS[method]

    for params in tqdm(list(param_grid), desc=f"{method} grid search"):
        correct = 0
        for i, item in enumerate(data):
            parsed = [r.get("parsed_answer") for r in item["responses"]]
            true_answer = str(item["true_answer"]).strip()

            try:
                labels = detect_fn(all_embeddings[i], params)
            except Exception:
                labels = np.ones(len(all_embeddings[i]))

            filtered = [parsed[j] for j in range(len(parsed)) if labels[j] == 1]
            answer = majority_vote(filtered) if filtered else majority_vote(parsed)

            if answer is not None and str(answer).strip().lower() == true_answer.lower():
                correct += 1

        acc = correct / len(data) * 100
        if acc > best_acc:
            best_acc = acc
            best_params = dict(params)

        print(f"  {dict(params)} => {acc:.2f}%")

    print(f"\n{'='*70}")
    print(f"Best {method}: {best_acc:.2f}% (Δ {best_acc - mv_acc:+.2f}%)")
    print(f"Best params: {best_params}")
    print(f"{'='*70}")

    return {"method": method, "best_accuracy": best_acc, "best_params": best_params, "majority_vote": mv_acc}


# ── Embedder selection by dataset ───────────────────────────────────────────

EMBEDDER_MAP = {
    "aqua": "allenai/scibert_scivocab_uncased",
    "svamp": "allenai/scibert_scivocab_uncased",
    "strategyqa": "roberta-base",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Outlier detection grid search")
    parser.add_argument("--json_path", required=True, help="Path to generated responses JSON")
    parser.add_argument("--dataset", required=True, choices=["aqua", "svamp", "strategyqa"])
    parser.add_argument("--method", default="all",
                        choices=["isolation_forest", "knn", "svm", "all"])
    parser.add_argument("--embedder", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    embedder = args.embedder or EMBEDDER_MAP[args.dataset]

    methods = ["isolation_forest", "knn", "svm"] if args.method == "all" else [args.method]
    results = {}

    for method in methods:
        result = run_gridsearch(args.json_path, embedder, args.dataset, method)
        results[method] = result

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for method, res in results.items():
        print(f"  {method:20s}: {res['best_accuracy']:.2f}% (Δ {res['best_accuracy'] - res['majority_vote']:+.2f}%)")


if __name__ == "__main__":
    main()
