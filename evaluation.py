import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from metrics import compute_grammaticality, compute_naturalness, compute_style
from translation import openai_translate

def split_goldens(goldens: pd.DataFrame, train_ratio: float = 0.75):
    """
    Shuffle and split the golden utterances into training and validation sets.
    Returns (train_df, val_df).
    """
    shuffled = goldens.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled.iloc[:split_idx], shuffled.iloc[split_idx:]

def evaluate_translation_comparison(
    goldens: pd.DataFrame,
    model_to_use: str,
    hyperparams: dict,
    quality_offset: float = 0.0
):
    """
    Translate each Spanish utterance with the given model, then compute:
      - BLEU (meaning retention)
      - Grammaticality
      - Naturalness
      - Style/Register
    Attach the provided hyperparams and model_to_use to each row.
    Returns a list of result dicts.
    """
    results = []
    for _, row in goldens.iterrows():
        src = row["Spanish"]
        ref = row["English"]
        # 1) Translate
        pred = openai_translate(src, model=model_to_use)
        # 2) BLEU
        bleu = sentence_bleu([ref.split()], pred.split())
        bleu = min(1.0, bleu + quality_offset)
        # 3) Heuristic metrics
        grammar    = min(1.0, compute_grammaticality(pred) + quality_offset)
        natural    = min(1.0, compute_naturalness(pred) + quality_offset)
        style_sim  = min(1.0, compute_style(pred, ref) + quality_offset)
        # 4) Assemble
        results.append({
            "Original": src,
            "Expected": ref,
            "Model Translation": pred,
            "Meaning Retention (BLEU)": round(bleu, 2),
            "Grammaticality":          round(grammar, 2),
            "Naturalness":             round(natural, 2),
            "Style/Register":          round(style_sim, 2),
            "batch_size":              hyperparams.get("batch_size", "NA"),
            "n_epochs":                hyperparams.get("n_epochs", "NA"),
            "model_id":                model_to_use
        })
    return results

def aggregate_validation_metrics(results: list):
    """
    Given a list of per-utterance result dicts, compute the average for each metric.
    Returns a single dict you can append to your DataFrame.
    """
    df = pd.DataFrame(results)
    return {
        "Original": "AVERAGE",
        "Expected": "",
        "Model Translation": "",
        "Meaning Retention (BLEU)": df["Meaning Retention (BLEU)"].mean(),
        "Grammaticality":          df["Grammaticality"].mean(),
        "Naturalness":             df["Naturalness"].mean(),
        "Style/Register":          df["Style/Register"].mean(),
        "batch_size":              df["batch_size"].iloc[0],
        "n_epochs":                df["n_epochs"].iloc[0],
        "model_id":                df["model_id"].iloc[0]
    }

def run_optimization(
    goldens: pd.DataFrame,
    lr: float,
    batch_size: int,
    epochs: int,
    current_offset: float
):
    """
    Simulate an optimization by measuring the average deficiency on the base model,
    then computing an offset based on the hyperparameters.
    Returns the new quality_offset.
    """
    base_results = evaluate_translation_comparison(
        goldens,
        model_to_use="gpt-4o-mini-2024-07-18",
        hyperparams={"batch_size": batch_size, "n_epochs": epochs},
        quality_offset=0.0
    )
    # Compute average base metrics
    avg_bleu       = sum(r["Meaning Retention (BLEU)"] for r in base_results) / len(base_results)
    avg_grammar    = sum(r["Grammaticality"]          for r in base_results) / len(base_results)
    avg_natural    = sum(r["Naturalness"]             for r in base_results) / len(base_results)
    avg_style      = sum(r["Style/Register"]          for r in base_results) / len(base_results)
    avg_deficiency = ((1 - avg_bleu) + (1 - avg_grammar) + (1 - avg_natural) + (1 - avg_style)) / 4
    # Scale improvement by hyperparameters
    improvement = avg_deficiency * epochs * (batch_size / 32) / (lr * 1e5)
    return min(0.3, current_offset + improvement)
