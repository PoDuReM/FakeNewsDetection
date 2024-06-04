import dataclasses

import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from vkr.ml import metrics
from vkr.utils import tqdm

from IPython.display import display
from matplotlib import pyplot as plt


@dataclasses.dataclass
class TrainResult:
    metrics_df: pd.DataFrame
    conf_matrices: list[list[tuple[str, np.ndarray]]]


def display_metrics_df(metrics_df: pd.DataFrame, columns: list[str]) -> None:
    phase_dfs = {}
    for phase in metrics_df['Phase'].unique():
        phase_df = metrics_df[metrics_df['Phase'] == phase].pivot(index='Epoch', columns='Phase',
                                                                  values=columns)
        phase_df.index.name = 'Эпоха'
        phase_df.columns = [f'{phase}_{col}' for col in phase_df.columns.levels[0]]
        phase_dfs[phase] = phase_df

    final_df = pd.concat(phase_dfs.values(), axis=1)
    new_columns = []
    metrics_translation = {
        'Accuracy': 'Точность',
        'F1-Score': 'F1-мера',
        'Precision': 'Прецизионность',
        'Recall': 'Полнота',
        'ROC AUC': 'ROC AUC'
    }
    for phase in phase_dfs.keys():
        for col in columns:
            new_columns.append(('', phase, metrics_translation.get(col, col)))

    final_df.columns = pd.MultiIndex.from_tuples(new_columns, names=['', 'Датасет', 'Метрика'])
    final_df[final_df.columns[final_df.columns.get_level_values(
        'Метрика') == 'Loss']] *= -1  # Invert Loss values for color mapping

    cmap = sns.diverging_palette(5, 150, as_cmap=True)
    formats = {col: '{:.2f}' for col in final_df.columns}
    for col in final_df.columns:
        if col[2] == 'Loss':
            formats[col] = lambda x: f'{-x:.2f}'
    table_styles = [
        {'selector': f'th.col{i}', 'props': [('border-right', '2px solid black')]}
        for i in range(len(final_df.columns) - 1) if (i + 1) % len(columns) == 0
    ]
    styled = final_df.style.set_table_styles(table_styles, overwrite=False).background_gradient(
        cmap=cmap).format(formats)

    display(styled)


def plot_loss(metrics_df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for phase in metrics_df['Phase'].unique():
        phase_data = metrics_df[metrics_df['Phase'] == phase]
        ax1.plot(phase_data['Epoch'], phase_data['Loss'], label=f'{phase}')
        ax2.plot(phase_data['Epoch'], phase_data['Loss'], label=f'{phase}')

    # Original scale
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.set_title('Потери по эпохам')
    ax1.legend()
    ax1.grid(True)

    # Log scale
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.set_yscale('log')
    ax2.set_title('Потери по эпохам (логарифмическая шкала)')
    ax2.legend()
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(conf_matrix: np.ndarray, labels: list[str], name: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font size for better readability
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels,
                yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{name} Confusion Matrix')
    plt.show()


def train_binary(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loaders: list[tuple[str, DataLoader]],
        optimizer,
        criterion,
        num_epochs: int,
        device: str,
) -> TrainResult:
    metrics_columns = [
        'Epoch', 'Phase', 'Loss', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC AUC',
    ]
    metrics_df = pd.DataFrame(columns=metrics_columns)
    conf_matrices = []

    # TODO: select best model on test metrics
    for epoch in range(num_epochs):
        print(f'{"-" * 75} Epoch {epoch + 1} {"-" * 75}')
        model.train()
        current_conf_matrices = []
        train_loss = 0
        for batch in tqdm.tqdm(train_loader, desc=f'Training, epoch={epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask).squeeze(-1)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Average train loss for the epoch
        train_result = metrics.evaluate(model, train_loader, device, criterion)
        train_metrics = metrics.calculate_metrics([train_result])
        current_conf_matrices.append(('Train', train_metrics.confusion_matrix))
        plot_confusion_matrix(train_metrics.confusion_matrix, ['FAKE', 'REAL'], 'Train')
        train_results = {
            'Epoch': epoch + 1, 'Phase': 'Train', 'Loss': train_metrics.loss,
            'Accuracy': train_metrics.accuracy, 'F1-Score': train_metrics.f1_score,
            'Precision': train_metrics.precision_score, 'Recall': train_metrics.recall_score,
            'ROC AUC': train_metrics.auc_score,
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([train_results])], ignore_index=True)

        cache_results = []
        for phase_name, val_loader in val_loaders:
            val_result = metrics.evaluate(model, val_loader, device, criterion)
            cache_results.append(val_result)

        # Evaluate on test set
        test_metrics = metrics.calculate_metrics(cache_results)
        current_conf_matrices.append(('Test', test_metrics.confusion_matrix))
        plot_confusion_matrix(test_metrics.confusion_matrix, ['FAKE', 'REAL'], 'Test')
        test_results = {
            'Epoch': epoch + 1, 'Phase': 'Test', 'Loss': test_metrics.loss,
            'Accuracy': test_metrics.accuracy, 'F1-Score': test_metrics.f1_score,
            'Precision': test_metrics.precision_score, 'Recall': test_metrics.recall_score,
            'ROC AUC': test_metrics.auc_score,
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([test_results])], ignore_index=True)
        for i in range(len(cache_results)):
            val_metrics = metrics.calculate_metrics([cache_results[i]])
            current_conf_matrices.append((val_loaders[i][0], val_metrics.confusion_matrix))
            plot_confusion_matrix(val_metrics.confusion_matrix, ['FAKE', 'REAL'],
                                  val_loaders[i][0])
            val_results = {
                'Epoch': epoch + 1, 'Phase': f'{val_loaders[i][0]}', 'Loss': val_metrics.loss,
                'Accuracy': val_metrics.accuracy, 'F1-Score': val_metrics.f1_score,
                'Precision': val_metrics.precision_score, 'Recall': val_metrics.recall_score,
                'ROC AUC': val_metrics.auc_score,
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([val_results])], ignore_index=True)

        conf_matrices.append(current_conf_matrices)
        display_metrics_df(metrics_df,
                           ['Loss', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC AUC'])
        if epoch > 0:
            plot_loss(metrics_df)

    return TrainResult(metrics_df=metrics_df, conf_matrices=conf_matrices)
