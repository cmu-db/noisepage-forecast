from preprocessor import Preprocessor
import pandas as pd
import matplotlib.pyplot as plt
import joypy
from tqdm import tqdm
import warnings
import pickle

class DataPreprocessor:
    def __init__(self, pred_interval: pd.Timedelta = pd.Timedelta("2S"), seq_length: int = 5,
                 pred_horizon: pd.Timedelta = pd.Timedelta("2S")):
        """
        Initialize the DataPreprocessor class with prediction interval, prediction length, and prediction horizon.

        @param pred_interval: the granularity by which the data points are aggregated
        @param seq_length: the length of the time series used to train the model
        @param pred_horizon: how far into the future the model predicts (model predicts t + pred_horizon)
        @precondition pred_interval should be equal to pred_horizon; otherwise, the prediction can be inaccurate
        """

        """
        The following params_dataframe have the shape of (T, N) where T is the number of timestamps and 
        N is the number of parameters. The column index is the timestamp and the row index is the parameter index.
        """
        # {key=template, value=params_dataframe}
        self.qt_to_original_df = {}
        # {key=template, value=params_dataframe}
        self.qt_to_normalized_df = {}

        # {key=template, value=[p1_dtype, p2_dtype, ...]}
        self.qt_to_dtype = {}
        # {key=template, value=[(p1_mean, p1_var), (p2_mean, p2_var), ...]}
        self.qt_to_stats = {}

        self.prediction_interval = pred_interval    # Each interval has two seconds
        self.sequence_length = seq_length           # 5 data points
        self.prediction_horizon = pred_horizon      # total time = interval * seq_len

        if pred_horizon != pred_interval:
            print(f"Warning: Prediction horizon {pred_horizon} is not equal to {pred_interval}.")

    def _get_param_data(self, query_df):
        """
        Generate directories that maps a query template to a dataframes and stats that contain parameter distributions.

        Each dataframe (template_series and normalized_template_series) has shape (T, N) where
        T is the number of timestamps and N is the number of parameters for this template.

        @param query_df: dataframe for all query template.
        """

        with warnings.catch_warnings():
            for index, (query_template, template_df) in enumerate(tqdm(query_df.groupby("query_template"))):

                # Skip query templates with no parameters ex. BEGIN
                if template_df["query_params"][0] == ():
                    continue

                # Extract param columns and strip off the quotation marks
                template_series = template_df["query_params"].apply(pd.Series)
                template_series = template_series.apply(lambda col: col.str.strip("\"'"))

                # replace the NaN values with empty string
                # note: this treatment might change the behavior in lines 72 - 85
                template_series.dropna(axis=0)


                dtypes = []
                stats = []

                # note: a bit awkward, but works for now
                for j, col in enumerate(template_series):
                    try:
                        template_series[col] = pd.to_numeric(template_series[col], errors="raise")
                        dtypes.append("numerical")
                    except:
                        try:
                            template_series[col] = pd.to_datetime(template_series[col], errors="raise")
                            dtypes.append("date")
                        except:
                            # todo: string columns are not considered for now
                            dtypes.append("string")

                # Make a copy of template_series to store normalized version
                # note: it is unnecessary to make the copy, in fact it might not be necessary to standardize
                normalized_template_series = template_series.copy(deep=True)
                template_series = template_series.convert_dtypes()
                for j, col in enumerate(normalized_template_series):
                    if dtypes[j] != "string":
                        mean = template_series[col].mean()
                        std = template_series[col].std()
                        if std != 0:
                            normalized_template_series[col] = (normalized_template_series[col] - mean) / std
                        else:
                            normalized_template_series[col] = normalized_template_series[col] - mean
                        stats.append((mean, std))
                    else:
                        stats.append(None)

                # Store original and standardized dataframe, dtype, and stats for this template
                self.qt_to_original_df[query_template] = template_series
                self.qt_to_normalized_df[query_template] = normalized_template_series
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
                    print(f"PARAM ${i + 1}")
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
    query_templates = list(dp.qt_to_original_df.keys())
    print("query template:", query_templates[0])
