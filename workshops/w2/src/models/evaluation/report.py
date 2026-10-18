"""Complete evaluation report for the single w2 model."""

from pathlib import Path

from ..lstm.config import EVALUATION_DIR, MODEL_ID
from .confusion import save_confusion
from .curves import plot_reliability_histogram, plot_roc_pr
from .figure_utils import show_or_close
from .optuna_history import plot_optuna_history
from .reports import save_classification_report
from .selection import save_selection_manifest
from .tables import DISPLAY_COLUMNS, build_metrics_table, save_metrics_tables


def _model_info(test_result, tune_result, data):
    model = test_result.get("model")
    return {
        "family": "lstm",
        "input": "sequence",
        "n_channels": data.input_dim,
        "n_parameters": (
            int(sum(parameter.numel() for parameter in model.parameters()))
            if model is not None else "N/A"
        ),
        "best_epoch": tune_result.get("best_epoch", "N/A"),
        "best_params": tune_result.get("best_params", "N/A"),
        "trial_number": tune_result.get("trial_number", "N/A"),
    }


def _metrics_and_tables(
    y_tune,
    y_test,
    proba_test,
    threshold,
    tune_result,
    model_id,
    model_info,
    out_dir,
):
    main, at_half = build_metrics_table(
        y_tune,
        y_test,
        proba_test,
        threshold,
        tune_result["best_value"],
        model_id,
        model_info,
    )
    table_paths = save_metrics_tables(main, at_half, out_dir)
    return main, at_half, table_paths


def _classification_outputs(
    y_test, proba_test, threshold, model_id, out_dir, show_plots
):
    report_dir = out_dir / "metrics" / "classification_reports"
    _, report_json, report_text = save_classification_report(
        y_test, proba_test, threshold, model_id, report_dir
    )
    fig, confusion_png, confusion_svg = save_confusion(
        y_test,
        proba_test,
        threshold,
        model_id,
        out_dir / "figures" / "confusion",
        tag="test",
    )
    show_or_close(fig, show_plots)
    return report_json, report_text, confusion_png, confusion_svg


def _curve_outputs(y_test, proba_test, model_id, out_dir, show_plots):
    roc_path = plot_roc_pr(
        y_test, proba_test, model_id, out_dir, show_plots
    )
    calibration_path = plot_reliability_histogram(
        y_test, proba_test, model_id, out_dir, show_plots
    )
    return roc_path, calibration_path


def _manifest_and_history(
    tune_result, threshold, model_id, out_dir, show_plots
):
    manifest_path = save_selection_manifest(
        model_id,
        threshold,
        tune_result["best_value"],
        tune_result.get("contract_hash", ""),
        out_dir,
    )
    study = tune_result.get("study")
    history_path = (
        plot_optuna_history(study, model_id, out_dir, show_plots)
        if study is not None else None
    )
    return manifest_path, history_path


def _paths(
    table_paths,
    report_json,
    report_text,
    confusion_png,
    confusion_svg,
    roc_path,
    calibration_path,
    manifest_path,
    optuna_path,
):
    return {
        "metrics": table_paths[0],
        "metrics_at_05": table_paths[1],
        "report_json": report_json,
        "report_text": report_text,
        "confusion_png": confusion_png,
        "confusion_svg": confusion_svg,
        "roc_pr": roc_path,
        "calibration": calibration_path,
        "selection_manifest": manifest_path,
        "optuna_history": optuna_path,
    }


def evaluate_model(
    test_result,
    tune_result,
    data,
    *,
    model_id=MODEL_ID,
    out_dir=EVALUATION_DIR,
    show_plots=True,
):
    """Write metrics, reports, confusion matrix, curves and the tune manifest."""
    out_dir = Path(out_dir)
    y_tune = data.labels("tune")
    y_test = test_result["y_test"]
    proba_test = test_result["proba_test"]
    threshold = tune_result["threshold"]

    main, at_half, table_paths = _metrics_and_tables(
        y_tune,
        y_test,
        proba_test,
        threshold,
        tune_result,
        model_id,
        _model_info(test_result, tune_result, data),
        out_dir,
    )
    report_json, report_text, confusion_png, confusion_svg = (
        _classification_outputs(
            y_test, proba_test, threshold, model_id, out_dir, show_plots
        )
    )
    roc_path, calibration_path = _curve_outputs(
        y_test, proba_test, model_id, out_dir, show_plots
    )
    manifest_path, optuna_path = _manifest_and_history(
        tune_result, threshold, model_id, out_dir, show_plots
    )

    paths = _paths(
        table_paths,
        report_json,
        report_text,
        confusion_png,
        confusion_svg,
        roc_path,
        calibration_path,
        manifest_path,
        optuna_path,
    )
    return {
        "metrics": main,
        "metrics_at_05": at_half,
        "display": main[DISPLAY_COLUMNS].round(5),
        "paths": paths,
    }
