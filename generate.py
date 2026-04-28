"""
Generate CoT responses for AQuA-RAT, SVAMP, and StrategyQA using OpenAI models.
Samples n responses per question at temperature=0.8, and also a greedy (t=0) baseline.
"""

import os
import re
import json
import argparse
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Few-shot CoT prompts (Wei et al. 2023 / Wang et al. 2023) ──────────────

AQUA_PROMPT = """Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? Answer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64
A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. The answer is (A).

Q: If a / b = 3/4 and 8a + 5b = 22,then find the value of a. Answer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2
A: If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2. The answer is (B).

Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km
A: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. The answer is (E).

Q: How many keystrokes are needed to type the numbers from 1 to 500? Answer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788
A: There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392. The answer is (B)."""

SVAMP_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."""

STRATEGYQA_PROMPT = """Q: Do hamsters provide food for any animals?
A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. The answer is yes.

Q: Could Brooke Shields succeed at University of Pennsylvania?
A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. The answer is yes.

Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?
A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. The answer is no.

Q: Yes or no: Is it common to see frost during some college commencements?
A: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. The answer is yes.

Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?
A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. The answer is no.

Q: Yes or no: Would a pear sink in water?
A: The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. The answer is no."""

PROMPTS = {
    "aqua": AQUA_PROMPT,
    "svamp": SVAMP_PROMPT,
    "strategyqa": STRATEGYQA_PROMPT,
}

# ── Dataset loading ─────────────────────────────────────────────────────────

def load_aqua(data_dir="data"):
    """Load AQuA-RAT test set (254 examples). Downloads if not present."""
    path = os.path.join(data_dir, "aqua_test.jsonl")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/shizhediao/active-prompt/main/dataset/AQuA/test.json"
        urllib.request.urlretrieve(url, path)
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            options_str = " ".join(item["options"])
            data.append({
                "question": f"{item['question']} Answer Choices: {options_str}",
                "true_answer": item["correct"],  # e.g. "A", "B", ...
            })
    return data


def load_svamp(data_dir="data"):
    """Load SVAMP (1000 examples). Downloads if not present."""
    path = os.path.join(data_dir, "svamp.json")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/shizhediao/active-prompt/main/dataset/SVAMP/SVAMP.json"
        urllib.request.urlretrieve(url, path)
    with open(path, "r") as f:
        raw = json.load(f)
    data = []
    for item in raw:
        question = f"{item['Body']} {item['Question']}"
        data.append({
            "question": question,
            "true_answer": str(item["Answer"]),
        })
    return data


def load_strategyqa(data_dir="data"):
    """Load StrategyQA train set (2290 examples, test labels not public)."""
    path = os.path.join(data_dir, "strategyqa_train.json")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/shizhediao/active-prompt/main/dataset/strategyQA/train.json"
        urllib.request.urlretrieve(url, path)
    with open(path, "r") as f:
        raw = json.load(f)
    data = []
    for item in raw:
        data.append({
            "question": item["question"],
            "true_answer": "yes" if item["answer"] else "no",
        })
    return data


DATASET_LOADERS = {
    "aqua": load_aqua,
    "svamp": load_svamp,
    "strategyqa": load_strategyqa,
}

# ── Answer extraction ───────────────────────────────────────────────────────

def extract_answer(response_text, dataset_name):
    """Parse the final answer from a CoT response."""
    text = response_text.strip().lower()

    if dataset_name == "aqua":
        # Look for "the answer is (X)" pattern
        match = re.search(r'the answer is\s*\(?([a-e])\)?', text)
        if match:
            return match.group(1).upper()
        # Fallback: last occurrence of a letter choice in parentheses
        matches = re.findall(r'\(([a-e])\)', text)
        return matches[-1].upper() if matches else None

    elif dataset_name == "svamp":
        # Look for "the answer is X" pattern
        match = re.search(r'the answer is\s*([\-\d\.]+)', text)
        if match:
            try:
                return str(int(float(match.group(1))))
            except ValueError:
                return match.group(1)
        # Fallback: last number in the response
        matches = re.findall(r'([\-]?\d+\.?\d*)', text)
        if matches:
            try:
                return str(int(float(matches[-1])))
            except ValueError:
                return matches[-1]
        return None

    elif dataset_name == "strategyqa":
        # Look for "the answer is yes/no"
        match = re.search(r'the answer is\s*(yes|no)', text)
        if match:
            return match.group(1)
        # Fallback: last yes/no in the response
        matches = re.findall(r'\b(yes|no)\b', text)
        return matches[-1] if matches else None

    return None

# ── Generation ──────────────────────────────────────────────────────────────

def generate_responses(model_name, prompt_with_question, n, temperature, top_p):
    """Generate n CoT responses from a given model."""
    responses = []
    for _ in range(n):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt_with_question}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=512,
        )
        responses.append(completion.choices[0].message.content)
    return responses


def run_generation(model_name, dataset_name, n=10, temperature=0.8, top_p=1.0,
                   output_dir="outputs", data_dir="data"):
    """Run full generation for one model × one dataset."""
    dataset = DATASET_LOADERS[dataset_name](data_dir)
    prompt = PROMPTS[dataset_name]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{dataset_name}_{model_name.replace('/', '-')}_n{n}_t{temperature}.json"
    )

    # Resume support: load existing results if any
    existing_results = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_results = json.load(f)
        print(f"Resuming from {len(existing_results)}/{len(dataset)} completed questions")

    results = existing_results
    for i, item in enumerate(tqdm(dataset, desc=f"{model_name}/{dataset_name}")):
        if i < len(existing_results):
            continue

        question = item["question"]
        input_prompt = f"{prompt}\n\nQ: {question}\nA:"
        raw_responses = generate_responses(model_name, input_prompt, n, temperature, top_p)

        parsed = []
        for resp in raw_responses:
            answer = extract_answer(resp, dataset_name)
            parsed.append({"response": resp, "parsed_answer": answer})

        result = {
            "question": question,
            "true_answer": item["true_answer"],
            "responses": parsed,
        }
        results.append(result)

        # Save after each question (for resume support)
        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {output_path}")
    return output_path


def run_greedy(model_name, dataset_name, output_dir="outputs", data_dir="data"):
    """Run greedy decoding baseline (n=1, t=0)."""
    return run_generation(
        model_name, dataset_name, n=1, temperature=0.0, top_p=1.0,
        output_dir=output_dir, data_dir=data_dir
    )

# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CoT responses for semantic self-consistency")
    parser.add_argument("--model", default="gpt-4o-mini", choices=["gpt-3.5-turbo", "gpt-4o-mini"],
                        help="OpenAI model to use")
    parser.add_argument("--dataset", default="aqua", choices=["aqua", "svamp", "strategyqa", "all"],
                        help="Dataset to evaluate on")
    parser.add_argument("--n", default=10, type=int, help="Number of sampled responses per question")
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--greedy", action="store_true", help="Also run greedy baseline (t=0, n=1)")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--data_dir", default="data")
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = ["aqua", "svamp", "strategyqa"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Generating: model={args.model}, dataset={ds}, n={args.n}, t={args.temperature}")
        print(f"{'='*60}")

        run_generation(
            args.model, ds, n=args.n, temperature=args.temperature,
            top_p=args.top_p, output_dir=args.output_dir, data_dir=args.data_dir
        )

        if args.greedy:
            print(f"\nRunning greedy baseline for {ds}...")
            run_greedy(args.model, ds, output_dir=args.output_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
