import dataclasses

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
import torch
from torch.utils.data import DataLoader

from vkr.utils import tqdm


@dataclasses.dataclass
class EvaluateResult:
    predictions: np.ndarray
    targets: np.ndarray
    total_loss: float


def evaluate(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
        criterion,
) -> EvaluateResult:
    model.eval()
    total_loss = 0
    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluate'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask).squeeze(-1)
            loss = criterion(logits, labels.float())
            total_loss += loss.item()

            preds = torch.sigmoid(logits).round().cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(batch['labels'].cpu().numpy())
    return EvaluateResult(
        predictions=np.array(predictions),
        targets=np.array(true_labels),
        total_loss=total_loss,
    )


@dataclasses.dataclass
class BinaryClassification:
    loss: float
    accuracy: float
    f1_score: float
    precision_score: float
    recall_score: float
    auc_score: float
    confusion_matrix: np.ndarray


def calculate_metrics(evaluate_results: list[EvaluateResult]) -> BinaryClassification:
    # Aggregate predictions and targets
    all_predictions = np.concatenate([result.predictions for result in evaluate_results])
    all_targets = np.concatenate([result.targets for result in evaluate_results])

    # Compute total loss
    total_loss = sum(result.total_loss for result in evaluate_results)
    average_loss = total_loss / len(all_targets)

    return BinaryClassification(
        loss=average_loss,
        accuracy=accuracy_score(all_targets, all_predictions),
        f1_score=f1_score(all_targets, all_predictions, average='binary'),
        precision_score=precision_score(all_targets, all_predictions),
        recall_score=recall_score(all_targets, all_predictions),
        auc_score=roc_auc_score(all_targets, all_predictions),
        confusion_matrix=confusion_matrix(all_targets, all_predictions)
    )
