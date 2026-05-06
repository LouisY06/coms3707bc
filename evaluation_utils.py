def normalize_answer(answer, dataset_name):
    """Normalize answers for dataset-specific exact-match evaluation."""
    if answer is None:
        return None
    text = str(answer).strip().lower()
    if dataset_name == "svamp":
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return str(int(number))
        return str(number)
    return text


def answers_match(predicted, true_answer, dataset_name):
    """Return whether a parsed prediction matches the dataset gold answer."""
    if predicted is None:
        return False
    return normalize_answer(predicted, dataset_name) == normalize_answer(true_answer, dataset_name)
