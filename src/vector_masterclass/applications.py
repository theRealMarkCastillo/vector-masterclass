"""Task-specific vector applications: explicit thresholds, exclusions, and abstention."""

import numpy as np

from .retrieval import normalized


def binary_metrics(expected, predicted):
    expected, predicted = np.asarray(expected, dtype=bool), np.asarray(predicted, dtype=bool)
    if expected.shape != predicted.shape or expected.ndim != 1 or not len(expected):
        raise ValueError("Require aligned nonempty one-dimensional binary labels")
    tp = int((expected & predicted).sum())
    fp = int((~expected & predicted).sum())
    fn = int((expected & ~predicted).sum())
    tn = int((~expected & ~predicted).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def route(vector, prototypes, labels, *, threshold=0.55, margin=0.05):
    if len(labels) != len(prototypes) or len(set(labels)) < 2:
        raise ValueError("Routing requires aligned prototypes for at least two intents")
    scores = normalized(prototypes) @ normalized(vector)
    per_label = {
        label: max(
            float(s) for candidate, s in zip(labels, scores, strict=True) if candidate == label
        )
        for label in set(labels)
    }
    ranked = sorted(per_label, key=lambda label: (-per_label[label], label))
    best, second = ranked[:2]
    accepted = per_label[best] >= threshold and per_label[best] - per_label[second] >= margin
    return {
        "intent": best if accepted else "abstain",
        "best_intent": best,
        "score": per_label[best],
        "margin": per_label[best] - per_label[second],
        "authorized": False,
    }


def neighbor_predict(query, vectors, labels, *, k=3, threshold=0.4):
    if not 1 <= k <= len(vectors) or len(vectors) != len(labels):
        raise ValueError("Invalid neighbor count or label alignment")
    scores = normalized(vectors) @ normalized(query)
    order = np.argsort(-scores, kind="stable")[:k]
    counts = {label: sum(labels[i] == label for i in order) for label in set(labels)}
    best = min(counts, key=lambda label: (-counts[label], label))
    return best if float(scores[order[0]]) >= threshold else "abstain"
