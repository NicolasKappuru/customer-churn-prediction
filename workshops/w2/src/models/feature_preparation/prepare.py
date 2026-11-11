"""Prepare the w2 sequence data for the PyTorch LSTM stage.

This is the only entry point needed before modeling. It loads the v5 export,
builds the shared fit/stop/tune/test split, fits the phase scalers and writes
the small artifacts that the model notebook will consume. The tensors remain in
their original ``.npy`` files and are read with ``mmap_mode="r"``.
"""

import argparse

from .artifacts import save_row_index, save_run_metadata
from .config import run_config, run_paths
from .data import load_cohorts, load_metadata, load_sequence_view
from .scaling import PhasePreprocessor, save_preprocessing
from .split import make_shared_split, validate_and_save_split


def _announce_paths(input_path, output_path, on_kaggle):
    """Print the resolved input/output paths."""
    print(
        "input:", input_path,
        "| output:", output_path,
        "| on_kaggle:", on_kaggle,
    )


def _load_inputs(input_path, config):
    """Load the contract, labels and memory-mapped sequences."""
    meta = load_metadata(input_path, config["expected_cutoffs"])
    print(
        "dataset_version:", meta["dataset_version"],
        "| cutoffs:", meta["cutoffs"],
    )
    cohort_df, y_all, cohort_labels = load_cohorts(
        input_path, meta, config["dev_cohort"], config["test_cohort"]
    )
    view = load_sequence_view(
        input_path, meta, config["dev_cohort"], config["test_cohort"]
    )
    print("stacked sequence view:", view.shape, view.dtype)
    return meta, cohort_df, y_all, cohort_labels, view


def _make_split(cohort_df, cohort_labels, y_all, meta, config, output_path):
    """Build and persist the shared split."""
    split, dev_rows = make_shared_split(
        cohort_df,
        cohort_labels,
        y_all,
        dev_cohort=config["dev_cohort"],
        test_cohort=config["test_cohort"],
        shares=config["split_shares"],
        hash_salt=config["split_hash_salt"],
    )
    summary, fit_rate = validate_and_save_split(
        split,
        dev_rows,
        cohort_labels,
        y_all,
        meta,
        output_path,
        dev_cohort=config["dev_cohort"],
        test_cohort=config["test_cohort"],
        expected_sizes=config["expected_full_sizes"],
    )
    print(summary.to_string(index=False))
    return split, dev_rows, summary, fit_rate


def _fit_phases(view, channel_specs, split, dev_rows, config, output_path):
    """Fit and save one scaler state per configured phase."""
    phase_rows = {
        "fit": split["fit"],
        "development": dev_rows,
    }
    preprocessor = PhasePreprocessor(
        view,
        channel_specs,
        phase_rows,
        quantile_sample=config["quantile_sample"],
        block_rows=config["block_rows"],
        seed=config["seed"],
    )
    phase_files = {}

    for phase in config["phases"]:
        state = preprocessor.state(phase)
        path = save_preprocessing(
            state, output_path / "preprocessing" / f"{phase}.json"
        )
        phase_files[phase] = path.relative_to(output_path).as_posix()
        print(
            f"fitted phase {phase!r}: rows={state['fit_rows']:,}"
            f" | file={phase_files[phase]}"
        )
    return preprocessor, phase_rows, phase_files


def _write_artifacts(
    cohort_df, config, meta, split, phase_rows, phase_files, fit_rate, output_path
):
    """Write the row index and the run metadata."""
    row_index = save_row_index(cohort_df, split, output_path)
    save_run_metadata(
        config,
        meta,
        split,
        phase_rows,
        phase_files,
        row_index,
        fit_rate,
        output_path,
    )
    return row_index


def prepare(input_dir=None, out_dir=None):
    """Run the feature-preparation stage and return the prepared objects."""
    config = run_config()
    on_kaggle, input_path, output_path = run_paths(
        input_dir=input_dir, out_dir=out_dir
    )
    _announce_paths(input_path, output_path, on_kaggle)

    meta, cohort_df, y_all, cohort_labels, view = _load_inputs(
        input_path, config
    )
    split, dev_rows, summary, fit_rate = _make_split(
        cohort_df, cohort_labels, y_all, meta, config, output_path
    )
    channel_specs = meta["sequences"]["lstm"]["channel_specs"]
    preprocessor, phase_rows, phase_files = _fit_phases(
        view, channel_specs, split, dev_rows, config, output_path
    )
    _write_artifacts(
        cohort_df, config, meta, split, phase_rows, phase_files, fit_rate, output_path
    )

    print("prepared artifacts:", output_path.resolve())
    return {
        "config": config,
        "meta": meta,
        "cohort_df": cohort_df,
        "y_all": y_all,
        "cohort_labels": cohort_labels,
        "view": view,
        "split": split,
        "dev_rows": dev_rows,
        "preprocessor": preprocessor,
        "summary": summary,
        "out_dir": output_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare w2 sequence data before LSTM training."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Sequence export directory (default: export-sequential-v5).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Artifact directory (default: artifacts/sherlock_w2/feature_preparation).",
    )
    args = parser.parse_args()
    prepare(input_dir=args.input_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
