import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = ['temperature', 'vibration', 'pressure']

    def fit(self, X, y=None):
        # We need to clean X first to fit the scaler properly, 
        # but the scaler expects clean numeric data.
        # So we clean a copy, fit scaler, then we are ready.
        X_clean = self._clean(X)
        self.scaler.fit(X_clean[self.feature_columns])
        return self

    def transform(self, X):
        X_clean = self._clean(X)
        X_scaled = X_clean.copy()
        X_scaled[self.feature_columns] = self.scaler.transform(X_clean[self.feature_columns])
        return X_scaled

    def _clean(self, X):
        df = X.copy()
        
        # 1. Force ALL feature columns to be numeric
        # This handles 'ERR', 'NULL', or empty strings in ANY column
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. Imputation (Missing Values)
        # Use a fixed default or the mean. 
        # Note: If df[col].mean() is NaN (all values are bad), we use 0.0 as a fallback.
        for col in self.feature_columns:
            if df[col].isnull().any():
                col_mean = df[col].mean()
                df[col] = df[col].fillna(col_mean if pd.notnull(col_mean) else 0.0)

        # 3. Outlier Removal / Clipping
        if 'vibration' in df.columns:
            df['vibration'] = df['vibration'].clip(upper=50)

        return df

if __name__ == "__main__":
    # Test
    df = pd.read_csv("raw_data.csv")
    processor = DataProcessor()
    processed_data = processor.fit_transform(df)
    print("Data processed successfully.")
    print(processed_data.head())
