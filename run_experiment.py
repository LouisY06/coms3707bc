"""
Run the full Semantic Self-Consistency experiment:
  1. Generate CoT responses (+ greedy baseline) for all model × dataset combos
  2. Evaluate with CPW, SCW, and outlier detection
  3. Print summary table
"""

import os
import json
import argparse

from generate import run_generation, run_greedy, SUPPORTED_MODELS
from evaluation_utils import answers_match

MODELS = ["gpt-4o-mini", "gpt-3.5-turbo", "claude-haiku-4-5-20251001", "gemini-2.0-flash"]
DATASETS = ["aqua", "svamp", "strategyqa"]

EMBEDDER_MAP = {
    "aqua": "allenai/scibert_scivocab_uncased",
    "svamp": "allenai/scibert_scivocab_uncased",
    "strategyqa": "roberta-base",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run full semantic self-consistency experiment")
    parser.add_argument("--step", default="all",
                        choices=["generate", "evaluate", "all"],
                        help="Which step to run (generate, evaluate, or all)")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=SUPPORTED_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--n", default=10, type=int)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--skip_outlier_gridsearch", action="store_true",
                        help="Skip the slow outlier detection grid search")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per dataset")
    return parser.parse_args()


def get_output_path(output_dir, dataset, model, n=10, t=0.8, limit=None):
    if limit is not None:
        return os.path.join(output_dir, f"{dataset}_{model.replace('/', '-')}_n{n}_t{t}_limit{limit}.json")
    return os.path.join(output_dir, f"{dataset}_{model.replace('/', '-')}_n{n}_t{t}.json")


def get_greedy_path(output_dir, dataset, model, limit=None):
    if limit is not None:
        return os.path.join(output_dir, f"{dataset}_{model.replace('/', '-')}_n1_t0.0_limit{limit}.json")
    return os.path.join(output_dir, f"{dataset}_{model.replace('/', '-')}_n1_t0.0.json")


def greedy_accuracy(json_path, dataset_name):
    """Compute accuracy for greedy decoding (single response)."""
    with open(json_path, "r") as f:
        data = json.load(f)
    correct = 0
    for item in data:
        parsed = item["responses"][0].get("parsed_answer")
        if answers_match(parsed, item["true_answer"], dataset_name):
            correct += 1
    return correct / len(data) * 100


def run_step_generate(args):
    """Step 1: Generate all CoT responses."""
    for model in args.models:
        for dataset in args.datasets:
            print(f"\n{'='*60}")
            print(f"GENERATING: {model} / {dataset} / n={args.n} / t=0.8")
            print(f"{'='*60}")
            run_generation(model, dataset, n=args.n, temperature=0.8,
                           output_dir=args.output_dir, data_dir=args.data_dir, limit=args.limit)

            print(f"\nGREEDY BASELINE: {model} / {dataset}")
            run_greedy(model, dataset, output_dir=args.output_dir, data_dir=args.data_dir, limit=args.limit)


def run_step_evaluate(args):
    """Step 2: Evaluate all methods."""
    from centroid_proximity_weighting import evaluate as cpw_evaluate
    from semantic_consensus_weighting import evaluate as scw_evaluate
    if not args.skip_outlier_gridsearch:
        from outlier_detection_gridsearch import run_gridsearch

    os.makedirs(args.results_dir, exist_ok=True)
    all_results = {}

    for model in args.models:
        for dataset in args.datasets:
            key = f"{model}/{dataset}"
            print(f"\n{'='*60}")
            print(f"EVALUATING: {key}")
            print(f"{'='*60}")

            sc_path = get_output_path(args.output_dir, dataset, model, args.n, limit=args.limit)
            greedy_path = get_greedy_path(args.output_dir, dataset, model, limit=args.limit)
            embedder = EMBEDDER_MAP[dataset]

            if not os.path.exists(sc_path):
                print(f"  SKIP (no generated file: {sc_path})")
                continue

            result = {"model": model, "dataset": dataset}

            # Greedy baseline
            if os.path.exists(greedy_path):
                result["greedy"] = greedy_accuracy(greedy_path, dataset)
                print(f"  Greedy: {result['greedy']:.2f}%")

            # CPW
            print(f"\n  Running CPW...")
            cpw_res = cpw_evaluate(sc_path, embedder, dataset)
            result["majority_vote"] = cpw_res["majority_vote"]
            result["cpw"] = cpw_res["cpw"]

            # SCW
            print(f"\n  Running SCW...")
            scw_res = scw_evaluate(sc_path, embedder, dataset)
            result["scw"] = scw_res["scw"]

            # Outlier detection
            if not args.skip_outlier_gridsearch:
                for method in ["isolation_forest", "knn", "svm"]:
                    print(f"\n  Running {method} grid search...")
                    od_res = run_gridsearch(sc_path, embedder, dataset, method)
                    result[f"outlier_{method}"] = od_res["best_accuracy"]
                    result[f"outlier_{method}_params"] = od_res["best_params"]

            all_results[key] = result

    # Save results
    results_path = os.path.join(args.results_dir, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print_summary(all_results)


def print_summary(all_results):
    """Print a summary table of all results."""
    print(f"\n{'='*90}")
    print(f"{'EXPERIMENT RESULTS':^90}")
    print(f"{'='*90}")

    header = f"{'Model/Dataset':<30} {'Greedy':>8} {'MajVote':>8} {'CPW':>8} {'SCW':>8} {'IF':>8} {'KNN':>8} {'SVM':>8}"
    print(header)
    print("-" * 90)

    for key, res in all_results.items():
        greedy = f"{res.get('greedy', 0):.1f}" if "greedy" in res else "—"
        mv = f"{res.get('majority_vote', 0):.1f}" if "majority_vote" in res else "—"
        cpw = f"{res.get('cpw', 0):.1f}" if "cpw" in res else "—"
        scw = f"{res.get('scw', 0):.1f}" if "scw" in res else "—"
        if_ = f"{res.get('outlier_isolation_forest', 0):.1f}" if "outlier_isolation_forest" in res else "—"
        knn = f"{res.get('outlier_knn', 0):.1f}" if "outlier_knn" in res else "—"
        svm = f"{res.get('outlier_svm', 0):.1f}" if "outlier_svm" in res else "—"

        print(f"{key:<30} {greedy:>8} {mv:>8} {cpw:>8} {scw:>8} {if_:>8} {knn:>8} {svm:>8}")

    print(f"{'='*90}")


def main():
    args = parse_args()

    if args.step in ("generate", "all"):
        run_step_generate(args)

    if args.step in ("evaluate", "all"):
        run_step_evaluate(args)


if __name__ == "__main__":
    main()
