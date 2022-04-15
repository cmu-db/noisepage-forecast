import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from scipy import stats
import joypy
import matplotlib.pyplot as plt

import pickle
import os
import warnings

from param_preprocessor import DataPreprocessor

from sklearn.model_selection import train_test_split

# LSTM config
HIDDEN_SIZE = 128
RNN_LAYERS = 2
EPOCHS = 50
LR = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
PREPROCESSOR_SAVE_PATH = "./data/data_preprocessor.pickle"
MODEL_SAVE_PATH = "./models/1m1p/"
qt_to_param_X_SAVE_PATH = "./data/qt_to_param_X.pickle"
qt_to_param_Y_SAVE_PATH = "./data/qt_to_param_Y.pickle"
qt_to_param_quantile_timeseries_SAVE_PATH = "./data/qt_to_param_quantile_timeseries.pickle"
qt_to_index_SAVE_PATH = "./data/qt_to_index.pickle"


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


class ParamQuantileData:
    """ 
    Contains train and test data for certain parameter
    """

    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def get_train_test_data(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test


class QueryQuantileData:
    """
    Contains train/test data for all parameters
    """

    def __init__(self, query_template, all_param_quantile_data):
        self.query_template = query_template
        self.param_quantile_data = all_param_quantile_data


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=HIDDEN_SIZE, num_layers=RNN_LAYERS):
        super(Network, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, dropout=0.1
        )
        self.classification = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=output_size),
        )

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        # output: (L, 1 * H_out)

        out = self.classification(output)
        return out


class Forecaster:
    def __init__(
        self, pred_interval=pd.Timedelta("2S"), pred_seq_len=5, pred_horizon=pd.Timedelta("2S"), load_metadata=False
    ):

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

        # Load DataPreprocessor class
        if load_metadata:
            with open(PREPROCESSOR_SAVE_PATH, "rb") as f:
                self.data_preprocessor = pickle.load(f)
        else:
            self.data_preprocessor = DataPreprocessor(pred_interval, pred_seq_len, pred_horizon,)

        # Prediction interval hyperparameters
        self.prediction_interval = pred_interval  # Each interval has two seconds
        self.prediction_seq_len = pred_seq_len  # 5 data points
        self.prediction_horizon = pred_horizon  # total time = interval * seq_len

        self.qt_to_param_X = {}
        self.qt_to_param_Y = {}
        self.qt_to_param_quantile_timeseries = {}
        self.qt_to_index = {}

        if load_metadata:
            self.load_forecast_metadata()

    def _generate_param_X_Y_dict(self):
        """qt_to_param_X: 
        key = query template
        value = [param1_quantile_df, param2_quantile_df, ...]
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # X has shape (N, seq_len, num_quantiles) --> each training instance
            # has shape (seq_len, num_quantiles)
            # Y has shape (N, num_quantiles)
            query_to_param_X = {}
            query_to_param_Y = {}
            for qt, tdfp in tqdm(self.data_preprocessor.qt_to_normalized_df.items()):
                param_X = []
                param_Y = []

                dtypes = self.data_preprocessor.qt_to_dtype[qt]
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

    def generate_time_series_data(self):
        """qt_to_param_quantile_timeseries:
        key = query template
        value = [(param1_X_train, param1_X_test, param1_Y_train, param1_Y_test), (param2...), ...]
        """
        self._generate_param_X_Y_dict()
        self.qt_to_param_quantile_timeseries = {}
        for qt in tqdm(self.qt_to_param_X.keys()):
            param_X = self.qt_to_param_X[qt]
            param_Y = self.qt_to_param_Y[qt]

            param_quantile_timeseries = []
            for i in range(len(param_X)):
                # Skip string parameter
                if self.data_preprocessor.qt_to_dtype[qt][i] == "string":
                    param_quantile_timeseries.append(None)
                    continue

                param_col_X = param_X[i]
                param_col_Y = param_Y[i]

                # Split into train test set
                X_train, X_test, Y_train, Y_test = train_test_split(
                    param_col_X, param_col_Y, shuffle=False, test_size=0.1
                )
                # Train on all data. TODO: This is just a hack right now to split
                # data into two portions. However, we still want to train on
                # all data.
                X_train, Y_train = param_col_X, param_col_Y

                # X: (N, L, H_in) to (L, N, H_in);
                X_train, X_test = np.transpose(X_train, (1, 0, 2)), np.transpose(X_test, (1, 0, 2))
                # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
                param_quantile_timeseries.append(ParamQuantileData(X_train, X_test, Y_train, Y_test))

            self.qt_to_param_quantile_timeseries[qt] = QueryQuantileData(qt, param_quantile_timeseries)
        # print(
        #     self.qt_to_param_quantile_timeseries[
        #         "DELETE FROM new_order WHERE NO_O_ID = $1 AND NO_D_ID = $2 AND NO_W_ID = $3"
        #     ]
        #     .param_quantile_data[0]
        #     .X_train.shape
        # )

    def save_checkpoint(ckpt_path, filename, query_template, model, epoch, optimizer, scheduler):
        path = os.path.join(ckpt_path, f"{filename}")

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        save_dict = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "qt": query_template,
        }

        if optimizer != None:
            save_dict["optimizer_state"] = optimizer.state_dict()
        if scheduler != None:
            save_dict["scheduler_state"] = scheduler.state_dict()

        torch.save(save_dict, path)
        # print(f"=> saved the model {filename} to {path}")

    def export_forecast_metadata(self):
        with open(qt_to_index_SAVE_PATH, "wb") as f:
            pickle.dump(self.qt_to_index, f)

        with open(qt_to_param_quantile_timeseries_SAVE_PATH, "wb") as f:
            pickle.dump(self.qt_to_param_quantile_timeseries, f)

        with open(qt_to_param_X_SAVE_PATH, "wb") as f:
            pickle.dump(self.qt_to_param_X, f)

        with open(qt_to_param_Y_SAVE_PATH, "wb") as f:
            pickle.dump(self.qt_to_param_Y, f)

    def load_forecast_metadata(self):
        with open(qt_to_index_SAVE_PATH, "rb") as f:
            self.qt_to_index = pickle.load(f)

        with open(qt_to_param_quantile_timeseries_SAVE_PATH, "rb") as f:
            self.qt_to_param_quantile_timeseries = pickle.load(f)

        with open(qt_to_param_X_SAVE_PATH, "rb") as f:
            self.qt_to_param_X = pickle.load(f)

        with open(qt_to_param_Y_SAVE_PATH, "rb") as f:
            self.qt_to_param_Y = pickle.load(f)

    ###################################################################################################
    #########################           Model Training        #########################################
    ###################################################################################################
    def _train_epoch(self, model, X_train, Y_train, optimizer, scheduler, loss_function):
        model.train()

        # Shuffle the timeseries
        arr = np.arange(X_train.shape[1])
        np.random.shuffle(arr)

        train_loss = 0
        batch_bar = tqdm(total=X_train.shape[1], dynamic_ncols=True, leave=False, position=0, desc="Train")
        for ind in arr:
            seq = torch.tensor(X_train[:, ind : ind + 1, :]).to(device).float()
            labels = torch.tensor(Y_train[ind]).to(device).float()
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred[-1, -1, :], labels)
            single_loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += float(single_loss)

            batch_bar.set_postfix(
                loss="{:.04f}".format(float(train_loss / (ind + 1))),
                lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
            )

            batch_bar.update()

        train_loss /= X_train.shape[1]
        batch_bar.close()
        return train_loss

    def _validate(self, model, X_test, Y_test, loss_function):
        # Validation loss
        model.eval()
        val_loss = 0
        batch_bar = tqdm(total=X_test.shape[1], dynamic_ncols=True, leave=False, position=0, desc="Validate")
        for ind in range(X_test.shape[1]):
            seq = torch.tensor(X_test[:, ind : ind + 1, :]).to(device).float()
            labels = torch.tensor(Y_test[ind]).to(device).float()

            with torch.no_grad():
                y_pred = model(seq)

            single_loss = loss_function(y_pred[-1, -1, :], labels)
            val_loss += float(single_loss)
            batch_bar.update()
        val_loss /= X_test.shape[1]
        batch_bar.close()
        return val_loss

    def _train_model(self):
        self.qt_to_index = {}
        for template_index, (qt, query_quantile_data_obj) in enumerate(self.qt_to_param_quantile_timeseries.items()):
            self.qt_to_index[qt] = template_index
            for param_index, param_quantile_data_obj in enumerate(query_quantile_data_obj.param_quantile_data):
                print(f"Training for Q{template_index}-P{param_index}...")

                # Skip string param
                if param_quantile_data_obj == None:
                    continue

                X_train, X_test, Y_train, Y_test = param_quantile_data_obj.get_train_test_data()

                model = Network(len(self.quantiles), len(self.quantiles), HIDDEN_SIZE, RNN_LAYERS).to(device)
                loss_function = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(X_train.shape[1] * EPOCHS))

                for epoch in range(EPOCHS):
                    train_loss = self._train_epoch(model, X_train, Y_train, optimizer, scheduler, loss_function)
                    val_loss = self._validate(model, X_test, Y_test, loss_function)

                # print(f"[LSTM FIT]epoch: {epoch + 1:3}, train_loss: {train_loss:10.8f}, val_loss: {val_loss:10.8f}")
                filename = f"{template_index}_{param_index}"
                self.save_checkpoint(MODEL_SAVE_PATH, filename, qt, model, epoch, None, None)

    def fit(self, log_filename, file_type="parquet", dataframe=None, save_metadata=True):
        print("Preprocessing data...")
        self.data_preprocessor.preprocess(log_filename, file_type, dataframe)
        print("Preprocessing Done!")
        if save_metadata:
            self.data_preprocessor.save_to_file(PREPROCESSOR_SAVE_PATH)

        print("Generating training data...")
        self.generate_time_series_data()
        print("Data generation done!")

        if save_metadata:
            self.export_forecast_metadata()

        # self._train_model()

    # Get all parameters for a query and compare it with actual data
    def get_all_parameters_for(self, query_template: str):
        template_original_df = self.data_preprocessor.qt_to_original_df[query_template]
        template_normalized_df = self.data_preprocessor.qt_to_normalized_df[query_template]
        template_dtypes = self.data_preprocessor.qt_to_dtype[query_template]
        template_stats = self.data_preprocessor.qt_to_stats[query_template]
        template_index = self.qt_to_index[query_template]

        for i, col in enumerate(template_normalized_df):
            # print(f"Processing parameter {i+1}...")
            # Skip non-numerical columns
            if template_dtypes[i] == "string":
                continue

            # Get corresponding model
            model = Network(len(self.quantiles), len(self.quantiles), HIDDEN_SIZE, RNN_LAYERS)
            filepath = os.path.join(MODEL_SAVE_PATH, f"{template_index}_{i}")
            state_dict = torch.load(filepath)
            model.load_state_dict(state_dict["model_state"])

            # Group by pred_iterval and get quantile data
            time_series_df = template_normalized_df[col].resample(self.prediction_interval).agg(self.quantiles)
            time_series_df = time_series_df.astype(float)

            # Get number of queries in each time interval
            num_template_df = template_normalized_df[col].resample(self.prediction_interval).count()
            # display(num_template_df.head())
            # display(time_series_df.head(10))

            # Build a new dataframe whichcontains predicted parameters for all timestamps
            generated_params = []
            timestamps = []
            for j in tqdm(range(len(time_series_df) - 1)):
                # Generate sequence data. Add padding if neccesary
                if j + 1 >= self.prediction_seq_len:
                    start_time = j - self.prediction_seq_len + 1
                    seq = time_series_df.iloc[start_time : (j + 1), :].to_numpy()
                else:
                    seq = time_series_df.iloc[: (j + 1), :].to_numpy()
                    seq = np.pad(seq, ((self.prediction_seq_len - j - 1, 0), (0, 0)))

                # Get predicted quantiles from the model
                seq = seq[None, :, :]
                seq = np.transpose(seq, (1, 0, 2))
                seq = torch.tensor(seq).to(device).float()
                with torch.no_grad():
                    pred = model(seq)

                # Ensure prediction quantile values are strictly increasing
                pred = pred[-1, -1, :]
                pred = torch.cummax(pred, dim=0).values

                # Generate num_template samples according to the distribution defined by the predicted quantile values
                pred = pred.cpu().detach().numpy()
                # print("pred:", pred)
                # print("actual:", time_series_df.iloc[j+1, :].to_numpy())
                # Un-normalize the quantiles
                mean, std = template_stats[i]
                if std != 0:
                    pred = pred * std + mean
                else:
                    pred = pred + mean

                class Dist(stats.rv_continuous):
                    def _cdf(self, x):
                        conditions = [x <= pred[0]]
                        for k in range(pred.shape[0] - 1):
                            conditions.append(pred[k] <= x <= pred[k + 1])
                        choices = [quantile_name / 100 for quantile_name in self.quantile_names]
                        return np.select(conditions, choices, default=0)

                dist = Dist(a=pred[0], b=pred[-1], name="deterministic")
                # Model takes in sequence data until time j and ouputs prediction value for time j+1.
                # Therefore we need the number of queries in thej+1's interval
                # num_templates = int(num_template_df[j+1]/10) # Divide by 10 so it runs faster
                num_params = 30  # Generate 30 parameter values for this specific parameter

                try:
                    for _ in range(num_params):
                        generated_params.append(int(dist.rvs()))
                        timestamps.append(num_template_df.index[j + 1])
                except:
                    # If all predicted quantiles have the same value, then a continous cdf cannot be constructed.
                    # Just take any predicted quantile value as the prediction.
                    for _ in range(num_params):
                        generated_params.append(pred[0])
                        timestamps.append(num_template_df.index[j + 1])

                # Generate a dataframe for the predicted parameter values
                predicted_params_df = pd.DataFrame(generated_params, index=pd.DatetimeIndex(timestamps))

                # Graph the results
                min_val, max_val = template_original_df[col].min(), template_original_df[col].max()
                min_val = min_val - (1 + 0.2 * (max_val - min_val))
                max_val = max_val + (1 + 0.2 * (max_val - min_val))

                print(f"PARAM ${i+1} Predicted")
                fig, axes = joypy.joyplot(
                    predicted_params_df.groupby(pd.Grouper(freq="5s")),
                    hist=True,
                    bins=20,
                    overlap=0,
                    grid=True,
                    x_range=[min_val, max_val],
                )
                plt.show()

                print(f"PARAM ${i+1} Actual")
                fig, axes2 = joypy.joyplot(
                    template_original_df[col].to_frame().groupby(pd.Grouper(freq="5s")),
                    hist=True,
                    bins=20,
                    overlap=0,
                    grid=True,
                    x_range=[min_val, max_val],
                )
                plt.show()
                print("\n")

    # Get all parameters for a query and compare it with actual data
    def get_parameters_for(self, query_template, timestamp, num_queries):
        target_timestamp = pd.Timestamp(timestamp)

        template_normalized_df = self.data_preprocessor.qt_to_normalized_df[query_template]
        template_dtypes = self.data_preprocessor.qt_to_dtype[query_template]
        template_stats = self.data_preprocessor.qt_to_stats[query_template]
        template_index = self.qt_to_index[query_template]
        param_X = self.qt_to_param_X[query_template]
        num_params = len(template_dtypes)

        generated_params = []

        for i in range(num_params):
            if template_dtypes[i] == "string":
                print("Skipping string columns")
                continue

            # Get corresponding model
            model = Network(len(self.quantiles), len(self.quantiles), HIDDEN_SIZE, RNN_LAYERS)
            filepath = os.path.join("./models/v1", f"{template_index}_{i}")
            state_dict = torch.load(filepath)
            model.load_state_dict(state_dict["model_state"])

            # Compute how many predictions need to be made
            start_timestamp = template_normalized_df.index.max()
            num_predictions = int((target_timestamp - start_timestamp) / self.prediction_interval)

            # Continously make predictions until target_timestamp
            param_X_col = param_X[i]
            seq = param_X_col[-1]
            seq = seq[None, :, :]
            seq = np.transpose(seq, (1, 0, 2))
            seq = torch.tensor(seq).to(device).float()
            for j in tqdm(range(num_predictions)):
                # Get predicted quantiles from the model
                with torch.no_grad():
                    pred = model(seq)

            # Ensure prediction quantile values are strictly increasing
            pred = pred[-1, -1, :]
            pred = torch.cummax(pred, dim=0).values

            # Add pred to original seq to create new seq for next time stamp
            seq = torch.squeeze(seq, axis=1)
            seq = torch.cat((seq[:-1, :], pred[None, :]), axis=0)
            seq = seq[:, None, :]

        pred = pred.cpu().detach().numpy()

        # Un-normalize the quantiles
        mean, std = template_stats[i]
        if std != 0:
            pred = pred * std + mean
        else:
            pred = pred + mean

        # Draw samples from the predicted distribution
        class Dist(stats.rv_continuous):
            def _cdf(self, x):
                conditions = [x <= pred[0]]
                for k in range(pred.shape[0] - 1):
                    conditions.append(pred[k] <= x <= pred[k + 1])
                choices = [quantile_name / 100 for quantile_name in self.quantile_names]
                return np.select(conditions, choices, default=0)

        dist = Dist(a=pred[0], b=pred[-1], name="deterministic")
        # Model takes in sequence data until time j and ouputs prediction value for time j+1.
        # Therefore we need the number of queries in thej+1's interval
        # num_templates = int(num_template_df[j+1]/10) # Divide by 10 so it runs faster
        # num_params = 30 # Generate 30 parameter values for this specific parameter

        generated_param_ith = []
        try:
            for _ in range(num_queries):
                generated_param_ith.append(dist.rvs())
        except:
            # If all predicted quantiles have the same value, then a continous cdf cannot be constructed.
            # Just take any predicted quantile value as the prediction.
            for _ in range(num_queries):
                generated_param_ith.append(pred[0])

        generated_params.append(generated_param_ith)
        return generated_params


if __name__ == "__main__":
    query_log_filename = "./preprocessed.parquet.gzip"

    forecaster = Forecaster(
        pred_interval=pd.Timedelta("2S"), pred_seq_len=5, pred_horizon=pd.Timedelta("2S"), load_metadata=False,
    )
    forecaster.fit(query_log_filename)

