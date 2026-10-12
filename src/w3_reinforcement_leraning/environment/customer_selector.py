"""Load and randomly select churned Telco customers."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


class CustomerSelector:
    """Keep a random sample of churned customers for the environment."""

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        sample_size: int = 1000,
    ) -> None:
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero.")

        self.dataset_path = Path(dataset_path).expanduser()
        if not self.dataset_path.is_absolute():
            self.dataset_path = PROJECT_ROOT / self.dataset_path
        self.sample_size = sample_size

        self.dataset = self.load_dataset()
        self.df = self.generate_customer_df()

    def load_dataset(self) -> pd.DataFrame:
        """Read and return the source Telco CSV."""
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Telco dataset not found: {self.dataset_path}")
        return pd.read_csv(self.dataset_path)

    def generate_customer_df(self) -> pd.DataFrame:


        """Create a random sample of churned customers without ID or target."""
        required_columns = {"customerID", "Churn"}
        missing_columns = required_columns.difference(self.dataset.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Telco dataset is missing required columns: {missing}")



        churned_customers = self.dataset.loc[self.dataset["Churn"].eq("Yes")].drop(
            columns=["customerID", "Churn"]
        )
        if len(churned_customers) < self.sample_size:
            raise ValueError(
                f"Requested {self.sample_size} churned customers, but only "
                f"{len(churned_customers)} are available."
            )
        
        print(f"Row number: {churned_customers.shape[0]}")


        return churned_customers.sample(
            n=self.sample_size,
            replace=False,
        ).reset_index(drop=True)

    def get_random_customer(self) -> pd.DataFrame:
        """Return one customer as a one-row DataFrame from the stored sample."""
        return self.df.sample(n=1).reset_index(drop=True)
