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


class DataPreprocessor:
    def __init__(
        self, pred_interval=pd.Timedelta("2S"), pred_seq_len=5, pred_horizon=pd.Timedelta("2S"),
    ):
        # {key=template, value=params_dataframe}
        self.qt_to_original_df = {}
        # {key=template, value=params_dataframe}
        self.qt_to_normalized_df = {}
        # {key=template, value=[p1_dtype, p2_dtype, ...]}
        self.qt_to_dtype = {}
        # {key=template, value=[(p1_mean, p1_var), (p2_mean, p2_var), ...]}
        self.qt_to_stats = {}

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

    def _clear_cache(self):
        self.qt_to_original_df = {}
        self.qt_to_normalized_df = {}
        self.qt_to_dtype = {}
        self.qt_to_stats = {}

    def preprocess(self, log_filename, file_type, dataframe=None):
        self._clear_cache()

        if file_type not in ["parquet", "csv", "dataframe"]:
            raise "File type must be parquet, csv, or pandas dataframe"
        if file_type == "parquet":
            # Get parsed query log
            preprocessor = Preprocessor(parquet_path=log_filename)
            df = preprocessor.get_dataframe()
            empties = df["query_template"] == ""
            print(f"Removing {sum(empties)} empty query template values.")
            df = df[:][~empties]
        elif file_type == "csv":
            # Get parsed query log
            preprocessor = Preprocessor(csvlogs=log_filename)
            df = preprocessor.get_dataframe()
            empties = df["query_template"] == ""
            print(f"Removing {sum(empties)} empty query template values.")
            df = df[:][~empties]
        elif file_type == "df":
            df = dataframe

        self._get_param_data(df)

    def save_to_file(self, file_path):
        if self.save_to_file:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)


if __name__ == "__main__":
    # query_log_filename = "./preprocessed.parquet.gzip"
    # dp = DataPreprocessor()
    # dp.preprocess(query_log_filename, "parquet")
    # dp.save_to_file("./data/data_preprocessor.pickle")

    # Load the class
    with open("./data/data_preprocessor.pickle", "rb") as f:
        dp = pickle.load(f)
        dp.graph_query_template(0)
    qts = list(dp.qt_to_original_df.keys())
    print("query template:", qts[0])
