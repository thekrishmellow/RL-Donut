"""The numbers a lesson reports back to the match page."""

from dataclasses import dataclass


@dataclass
class LearnReport:
    rows: int
    latest_rows: int
    epochs: int
    logprob_before: float
    logprob_after: float
    logprob_all_before: float
    logprob_all_after: float
    improved: bool

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "latest_rows": self.latest_rows,
            "epochs": self.epochs,
            "logprob_before": self.logprob_before,
            "logprob_after": self.logprob_after,
            "logprob_all_before": self.logprob_all_before,
            "logprob_all_after": self.logprob_all_after,
            "improved": self.improved,
        }
