import difflib
from typing import List

from utils import confidence_to_color


def compare_phonetic_strings(truth: str, predicted: str, confidences: List[float]):
    truth = truth.lower().strip()
    predicted = predicted.lower().strip()

    matcher = difflib.SequenceMatcher(None, truth, predicted)

    result = []
    pred_idx = 0

    total_chars = len(truth)
    correct_chars = 0
    substitutions = 0
    insertions = 0
    deletions = 0

    for op, t_start, t_end, p_start, p_end in matcher.get_opcodes():
        if op == "equal":
            for i in range(t_start, t_end):
                confidence = (
                    confidences[pred_idx] if pred_idx < len(confidences) else 1.0
                )
                result.append(
                    {
                        "char": truth[i],
                        "predicted": "",
                        "status": "correct",
                        "confidence": confidence,
                        "color": confidence_to_color(confidence),
                    }
                )
                pred_idx += 1
                if confidence > 0.6:
                    correct_chars += 1
        elif op == "replace":
            for i in range(t_start, t_end):
                if pred_idx < len(predicted):
                    confidence = (
                        confidences[pred_idx] if pred_idx < len(confidences) else 0.0
                    )
                    result.append(
                        {
                            "char": truth[i],
                            "predicted": predicted[pred_idx],
                            "status": "substitution",
                            "confidence": confidence,
                            "color": confidence_to_color(confidence // 3),
                        }
                    )
                    pred_idx += 1
                    substitutions += 1
                else:
                    result.append(
                        {
                            "char": truth[i],
                            "predicted": "",
                            "status": "deletion",
                            "confidence": 0.0,
                            "color": confidence_to_color(0.0),
                        }
                    )
                    deletions += 1
        elif op == "insert":
            for i in range(p_start, p_end):
                confidence = confidences[i] if i < len(confidences) else 0.0
                result.append(
                    {
                        "char": "",
                        "predicted": predicted[i],
                        "status": "insertion",
                        "confidence": confidence,
                        "color": confidence_to_color(confidence // 3),
                    }
                )
                insertions += 1
                pred_idx += 1
        elif op == "delete":
            for i in range(t_start, t_end):
                result.append(
                    {
                        "char": truth[i],
                        "predicted": "",
                        "status": "deletion",
                        "confidence": 0.0,
                        "color": confidence_to_color(0.0),
                    }
                )
                deletions += 1

    accuracy = correct_chars / total_chars if total_chars > 0 else 0
    error_rate = (
        (substitutions + insertions + deletions) / total_chars if total_chars > 0 else 1
    )

    return {
        "detailed_comparison": result,
        "metrics": {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "correct_chars": correct_chars,
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "total_chars": total_chars,
        },
    }
