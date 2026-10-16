"""Ordinal encoding using only each phase's training vocabulary."""

import numpy as np
import pandas as pd


def build_rf_vocabulary(frame, categorical):
    """Learn category levels from the supplied training frame."""
    return {
        column: sorted(pd.unique(frame[column].dropna()).tolist())
        for column in categorical
    }


def encode_rf_frame(frame, vocab, features, categorical):
    """Encode known levels as ordinal codes and unseen levels as -1."""
    encoded = frame[features].copy()
    for column in categorical:
        codes = {level: index for index, level in enumerate(vocab[column])}
        encoded[column] = encoded[column].map(codes).fillna(-1).astype("float64")
    return encoded


class RandomForestData:
    """Cache phase vocabularies and provide encoded train/test frames."""

    def __init__(self, train_df, test_df, phase_rows, features, categorical):
        self.train_df = train_df
        self.test_df = test_df
        self.phase_rows = phase_rows
        self.features = features
        self.categorical = categorical
        self._vocab = {}

    def vocab(self, phase):
        if phase not in self._vocab:
            rows = self.phase_rows[phase]
            self._vocab[phase] = build_rf_vocabulary(
                self.train_df.iloc[rows], self.categorical
            )
        return self._vocab[phase]

    def frame(self, rows, phase, dataset="train"):
        if dataset == "train":
            source = self.train_df
        elif dataset == "test":
            source = self.test_df
        else:
            raise ValueError(f"Unknown dataset: {dataset!r}")
        return encode_rf_frame(
            source.iloc[np.asarray(rows)],
            self.vocab(phase),
            self.features,
            self.categorical,
        )
