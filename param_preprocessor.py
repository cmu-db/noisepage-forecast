from sklearn.preprocessing import LabelEncoder
from preprocessor import Preprocessor

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joypy
from ast import literal_eval
from pandas.api.types import is_datetime64_any_dtype
from tqdm import tqdm

import warnings
import pickle

QUERY_LOG_FILENAME = "./preprocessed.parquet.gzip"


class QuantileMetadata:
    @classmethod
    def left_boundary(x):
        return x.quantile(0.01)

    def q1(x):
        return x.quantile(0.1)

    def q2(x):
        return x.quantile(0.2)

    def q3(x):
        return x.quantile(0.3)

    def q4(x):
        return x.quantile(0.4)

    def q5(x):
        return x.quantile(0.5)

    def q6(x):
        return x.quantile(0.6)

    def q7(x):
        return x.quantile(0.7)

    def q8(x):
        return x.quantile(0.8)

    def q9(x):
        return x.quantile(0.9)

    def right_boundary(x):
        return x.quantile(0.99)


class DataPreprocessor:
    def __init__(self, pred_interval=pd.Timedelta("2S"), pred_seq_len=5, pred_horizon=pd.Timedelta("2S")):
        # {key=template, value=params_dataframe}
        self.qt_to_original_df = {}
        # {key=template, value=params_dataframe}
        self.qt_to_normalized_df = {}
        # {key=template, value=[p1_dtype, p2_dtype, ...]}
        self.qt_to_dtype = {}
        # {key=template, value=[(p1_mean, p1_var), (p2_mean, p2_var), ...]}
        self.qt_to_stats = {}

        # Quantiles to be used to generate training data
        self.quantiles = [
            QuantileMetadata.left_boundary,
            QuantileMetadata.q1,
            QuantileMetadata.q2,
            QuantileMetadata.q3,
            QuantileMetadata.q4,
            QuantileMetadata.q5,
            QuantileMetadata.q6,
            QuantileMetadata.q7,
            QuantileMetadata.q8,
            QuantileMetadata.q9,
            QuantileMetadata.right_boundary,
        ]
        self.quantile_names = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Prediction interval hyperparameters
        self.prediction_interval = pred_interval  # Each interval has two seconds
        self.prediction_seq_len = pred_seq_len  # 5 data points
        self.prediction_horizon = pred_horizon  # total time = interval * seq_len

    def _get_param_data(self, df):
        """Generate a dictionary that maps a query template (qt) to a param
        dataframe.
        key=qt
        value= | TS | p1 | p2 | p3 | ...

        Args:
            df: query log dataframe
            save: whether export the generated dictionaries to pickle file
        """
        with warnings.catch_warnings():
            for index, (query_template, tdf) in enumerate(tqdm(df.groupby("query_template"))):

                # print(f"******************************************")
                # print(index, query_template)

                # Skip query templates with no parameters ex. BEGIN
                if tdf["query_params"][0] == ():
                    continue

                # Extract param columns and strip off the quotation marks
                tdfp = tdf["query_params"].apply(pd.Series)
                tdfp = tdfp.apply(lambda col: col.str.strip("\"'"))

                # Make a copy of tdfp to store normalized version
                normalized_tdfp = tdfp.copy(deep=True)

                dtypes = []
                stats = []
                for j, col in enumerate(tdfp):
                    # TODO: if a column contains numerical values and NAN, then need
                    # to fill the NAN values before doing this step.
                    try:
                        tdfp[col] = pd.to_numeric(tdfp[col], errors="raise")
                        normalized_tdfp[col] = pd.to_numeric(normalized_tdfp[col], errors="raise")
                        dtypes.append("numerical")
                    except:
                        try:
                            tdfp[col] = pd.to_datetime(tdfp[col], errors="raise")
                            normalized_tdfp[col] = pd.to_datetime(normalized_tdfp[col], errors="raise")
                            dtypes.append("date")
                        except:
                            # TODO: Right now we drop non date/numerical columns. Want to handle string columns later
                            dtypes.append("string")
                            pass

                    # Compute mean/var and standardize the column
                    if dtypes[-1] != "string":
                        # print(f"param {j}, {dtypes[-1]}")
                        mean = tdfp[col].mean()
                        std = tdfp[col].std()
                        # print(mean, std)
                        # print(tdfp[col])
                        if std != 0:
                            normalized_tdfp[col] = (normalized_tdfp[col] - mean) / std
                        else:
                            normalized_tdfp[col] = normalized_tdfp[col] - mean
                        stats.append((mean, std))
                    else:
                        stats.append(None)
                tdfp = tdfp.convert_dtypes()

                # Store df, dtype, and stats for this template
                self.qt_to_original_df[query_template] = tdfp
                self.qt_to_normalized_df[query_template] = normalized_tdfp
                self.qt_to_dtype[query_template] = dtypes
                self.qt_to_stats[query_template] = stats

    def graph_query_template(self, template_index, template_str=None):
        if template_str != None:
            qt = self.qt_to_original_df[template_str]
        else:
            qt = list(self.qt_to_original_df.keys())[template_index]

        gdft = self.qt_to_original_df[qt]
        print("Query:", qt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, col in enumerate(gdft):
                meow = gdft[col].to_frame()
                try:
                    # Compute x_lin
                    min_val, max_val = meow.min(), meow.max()
                    min_val = min_val - (1 + 0.2 * (max_val - min_val))
                    max_val = max_val + (1 + 0.2 * (max_val - min_val))
                    joypy.joyplot(
                        meow.groupby(pd.Grouper(freq="5s")),
                        hist=True,
                        bins=20,
                        overlap=0,
                        grid=True,
                        x_range=[min_val, max_val],
                    )
                    print(f"PARAM ${i+1}")
                    plt.show()
                except:
                    pass

    def generate_training_data(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # X has shape (N, seq_len, num_quantiles) --> each training instance has shape (seq_len, num_quantiles)
            # Y has shape (N, num_quantiles)
            query_to_param_X = {}
            query_to_param_Y = {}
            for qt, tdfp in tqdm(self.qt_to_normalized_df.items()):
                param_X = []
                param_Y = []

                dtypes = self.qt_to_dtype[qt]
                for j, col in enumerate(tdfp):
                    # Skip non-numerical columns
                    if dtypes[j] == "string":
                        param_X.append(None)
                        param_Y.append(None)
                        continue

                    param_col_X = []
                    param_col_Y = []
                    # Group by time and get quantile data
                    time_series_df = tdfp[col].resample(self.prediction_interval).agg(self.quantiles)
                    time_series_df = time_series_df.astype(float)
                    # display(time_series_df.head())
                    shifted = time_series_df.shift(freq=-self.prediction_horizon).reindex_like(time_series_df).ffill()

                    # Generate training instance. Add padding if neccesary
                    for i in range(len(time_series_df) - 1):
                        if i + 1 >= self.prediction_seq_len:
                            i_start = i - self.prediction_seq_len + 1
                            param_col_X.append(time_series_df.iloc[i_start : (i + 1), :].to_numpy())
                            param_col_Y.append(shifted.iloc[i, :].to_numpy())
                        else:
                            x = time_series_df.iloc[: (i + 1), :].to_numpy()
                            # Add padding above the rows
                            x = np.pad(x, ((self.prediction_seq_len - i - 1, 0), (0, 0)))
                            param_col_X.append(x)
                            param_col_Y.append(shifted.iloc[i, :].to_numpy())
                    param_col_X, param_col_Y = np.asarray(param_col_X), np.asarray(param_col_Y)
                    param_X.append(param_col_X)
                    param_Y.append(param_col_Y)

                query_to_param_X[qt] = param_X
                query_to_param_Y[qt] = param_Y
            self.qt_to_param_X = query_to_param_X
            self.qt_to_param_Y = query_to_param_Y

    def preprocess(self):
        # Get parsed query log
        preprocessor = Preprocessor(parquet_path=QUERY_LOG_FILENAME)
        df = preprocessor.get_dataframe()
        empties = df["query_template"] == ""
        print(f"Removing {sum(empties)} empty query template values.")
        df = df[:][~empties]
        self._get_param_data(df)
        self.generate_training_data()

    def save_to_file(self, file_path):
        if self.save_to_file:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)


if __name__ == "__main__":
    # dp = DataPreprocessor()
    # dp.preprocess()
    # dp.save_to_file("./data/data_preprocessor.pickle")

    # Load the class
    with open("./data/data_preprocessor.pickle", "rb") as f:
        dp2 = pickle.load(f)
        # dp2.graph_query_template(0)
    qts = list(dp2.qt_to_param_X.keys())
    print("query template:", qts[0])
    print("number of params:", len(dp2.qt_to_param_X[qts[0]]))
    print("training data shape for param 1:", dp2.qt_to_param_X[qts[0]][0].shape, dp2.qt_to_param_Y[qts[0]][0].shape)

