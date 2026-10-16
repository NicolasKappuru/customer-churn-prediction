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
    """Cache February vocabularies and encode labelled cohort rows."""

    def __init__(self, cohort_df, phase_rows, features, categorical):
        self.cohort_df = cohort_df
        self.phase_rows = phase_rows
        self.features = features
        self.categorical = categorical
        self._vocab = {}

    def vocab(self, phase):
        if phase not in self._vocab:
            rows = self.phase_rows[phase]
            self._vocab[phase] = build_rf_vocabulary(
                self.cohort_df.iloc[rows], self.categorical
            )
        return self._vocab[phase]

    def frame(self, rows, phase):
        return encode_rf_frame(
            self.cohort_df.iloc[np.asarray(rows)],
            self.vocab(phase),
            self.features,
            self.categorical,
        )
