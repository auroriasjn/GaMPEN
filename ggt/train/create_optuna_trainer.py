import mlflow
import torch.nn as nn
import torch
import torch.nn.functional as F
import optuna

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, Loss, Accuracy, Precision, Recall, ConfusionMatrix

from ggt.metrics import ElementwiseMae
from ggt.losses import AleatoricLoss, AleatoricCovLoss
from ggt.utils import (
    metric_output_transform_al_loss,
    metric_output_transform_al_cov_loss,
    plot_confusion_matrix,
)

def create_optuna_trainer(trial, model, optimizer, criterion, loaders, device):
    """Set up Ignite trainer and evaluator."""
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )

    if isinstance(criterion, AleatoricLoss):
        output_transform = metric_output_transform_al_loss
    elif isinstance(criterion, AleatoricCovLoss):
        output_transform = metric_output_transform_al_cov_loss
    else:
        def output_transform(x):
            return x

        def transform_binary(x):
            probabilities = F.softmax(x[0], dim=1)
            y_pred = torch.argmax(probabilities, dim=1)
            y = x[1]

            new_labels = (y_pred, y)
            return new_labels
        def transform_binary_onehot(x):
            y_pred, y = x
            y_pred = F.softmax(x[0], dim=1).round().long()
            y = y.long()

            return y_pred, y

    if isinstance(criterion, nn.CrossEntropyLoss) or isinstance(criterion, nn.NLLLoss) or isinstance(criterion, nn.BCEWithLogitsLoss):
        metrics = {
            "loss": Loss(criterion),
            "accuracy": Accuracy(output_transform=transform_binary),
            "precision": Precision(output_transform=transform_binary),
            "recall": Recall(output_transform=transform_binary),
            "cm": ConfusionMatrix(num_classes=2, output_transform=transform_binary_onehot)
        }
    else:
        metrics = {
            "mae": MeanAbsoluteError(output_transform=output_transform),
            "elementwise_mae": ElementwiseMae(output_transform=output_transform),
            "mse": MeanSquaredError(output_transform=output_transform),
            "loss": Loss(criterion)
        }

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    # Define training hooks
    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        for L, loader in loaders.items():
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            for M in metrics.keys():
                print(f'{M}: {metrics[M]}')
                if M == "elementwise_mae":
                    for i, val in enumerate(metrics[M].tolist()):
                        mlflow.log_metric(f"{L}-{M}-{i}", val, 0)
                elif M == "cm":
                    continue
                else:
                    mlflow.log_metric(f"{L}-{M}", metrics[M], 0)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_devel_results(trainer):
        evaluator.run(loaders["devel"])
        metrics = evaluator.state.metrics
        if trial.should_prune():
            raise optuna.TrialPruned()

        for M in metrics.keys():
            print(f'{M}: {metrics[M]}')
            if M == "elementwise_mae":
                for i, val in enumerate(metrics[M].tolist()):
                    mlflow.log_metric(
                        f"devel-{M}-{i}", val, trainer.state.epoch
                    )
            elif M == "cm":
                continue
            else:
                mlflow.log_metric(
                    f"devel-{M}", metrics[M], trainer.state.epoch
                )

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        for L, loader in loaders.items():
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            for M in metrics.keys():
                print(f'END: {M}: {metrics[M]}')
                if M == "elementwise_mae":
                    for i, val in enumerate(metrics[M].tolist()):
                        mlflow.log_metric(
                            f"{L}-{M}-{i}", val, trainer.state.epoch
                        )
                elif M == "cm":
                    # The confusion matrix is formatted such that columns are predictions and rows are targets
                    # Horiz is predicted, vert is actual
                    print(f'{L} CONFUSION MATRIX: {metrics[M]}')
                    plot_confusion_matrix(metrics[M], ['Non-Merger', 'Merger'], f'{L}-cm.png')
                    mlflow.log_artifact(f'{L}-cm.png')
                else:
                    mlflow.log_metric(
                        f"{L}-{M}", metrics[M], trainer.state.epoch
                    )

    return trainer
