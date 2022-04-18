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
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
PREPROCESSOR_SAVE_PATH = "./data/data_preprocessor.pickle"
MODEL_SAVE_PATH = "./models/1m1t/"
qt_to_param_X_SAVE_PATH = "./data/qt_to_param_X.pickle"
qt_to_param_Y_SAVE_PATH = "./data/qt_to_param_Y.pickle"
qt_to_param_quantile_timeseries_SAVE_PATH = "./data/qt_to_param_quantile_timeseries.pickle"
qt_to_index_SAVE_PATH = "./data/qt_to_index.pickle"
qt_to_num_params_SAVE_PATH = "./data/qt_to_num_params.pickle"

class ParamQuantileData:
    """ 
    Contains train and test data one parameter.
    The four member variables are all np list having the shape (num_samples, seq_length, num_quantiles + n),
    where n is the number of parameters in the template that the current parameter belongs to.
    """

    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def get_train_test_data(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test


class Network(nn.Module):
    """
    Simple LSTM model.
    todo: add an embedding layer
    """

    def __init__(self, input_size, output_size, hidden_size=HIDDEN_SIZE, num_layers=RNN_LAYERS):
        super(Network, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, dropout=0.1,
            batch_first=True
        )
        self.classification = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=output_size),
        )

    def forward(self, x, hidden):
        if hidden is None:
            output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden)
        prediction = self.classification(output)
        return prediction, hidden


# Draw samples from the predicted distribution
class Dist(stats.rv_continuous):
    def __init__(self, a, b, name, pred, quantiles):
        super(Dist, self).__init__(a=a, b=b, name=name)
        self.pred = pred
        self.quantiles = quantiles

    def _cdf(self, x):  # Rename self to self_dist so that self refers to outer class
        conditions = [x <= self.pred[0]]
        for k in range(self.pred.shape[0] - 1):
            conditions.append(self.pred[k] <= x <= self.pred[k + 1])
        choices = self.quantiles
        return np.select(conditions, choices, default=0)


class Forecaster:
    def __init__(
            self, pred_interval=pd.Timedelta("2S"), pred_seq_len=5, pred_horizon=pd.Timedelta("2S"),
            load_metadata=False):
        """
        Initialize Forecaster class.

        @param pred_interval:
        @param pred_seq_len:
        @param pred_horizon:
        @param load_metadata:
        """

        self.quantiles = np.arange(.01, 1, .01)

        # Load DataPreprocessor class
        if load_metadata:
            with open(PREPROCESSOR_SAVE_PATH, "rb") as f:
                self.data_preprocessor = pickle.load(f)
        else:
            self.data_preprocessor = DataPreprocessor(pred_interval, pred_seq_len, pred_horizon)

        # note: if the data is loaded from the pickle, the parameters should match

        # Prediction interval hyperparameters
        self.prediction_interval = pred_interval  # Each interval has two seconds
        self.prediction_seq_len = pred_seq_len  # 5 data points
        self.prediction_horizon = pred_horizon  # total time = interval * seq_len

        self.qt_to_param_X = {}
        self.qt_to_param_Y = {}
        self.qt_to_param_quantile_timeseries = {}
        self.qt_to_index = {}
        self.query_to_num_params = {}

        if load_metadata:
            self.load_forecast_metadata()

    def generate_time_series_data(self):
        """
        Split the normalized query template dataframe into matrices that can be fed into the LSTM.
        In the end, the query_to_param_X and query_to_param_Y dictionaries are constructed.
        The description of the dictionaries can be found below.
        The training and testing data are also generated here.

        Note that this is different from the original way of partitioning the data points.
        Specifically, X and Y have the same shape for a template's parameter. Y is simply shifted one step forward.
        X is also non-overlapping (i.e. 1st point: [t : t + seq_len], 2nd point [t + seq_len : t + 2seq_len]).
        There is no point to feed the LSTM with data points it has seen before.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # note: each template_df is in the shape of (T, N)
            # note: where T is the number of timestamps and N is the number of parameters
            for query_template, template_df in tqdm(self.data_preprocessor.qt_to_normalized_df.items()):
                # list of matrices for a given query_template
                param_X = []
                param_Y = []

                dtypes = self.data_preprocessor.qt_to_dtype[query_template]
                num_cols = len(template_df.columns)

                self.query_to_num_params[query_template] = num_cols

                param_quantile_timeseries = []

                for j, col in enumerate(template_df):
                    # Skip non-numerical columns
                    if dtypes[j] == "string":
                        param_X.append(None)
                        param_Y.append(None)
                        continue

                    # Group by time and get quantile data
                    # todo: should be fairly easy to tune the range in order to get more fine-grained distribution
                    # note: should also decide if it is legitimate to use 0.01 and 0.99 as the left and right boundary
                    func_quantiles = map(lambda y: lambda x: x.quantile(y), self.quantiles)
                    time_series_df = template_df[col].resample(self.prediction_interval)
                    time_series_df = time_series_df.agg(func_quantiles)

                    # note: add N columns to indicate which parameter current time series represents
                    for param_index in range(num_cols):
                        if param_index == j:
                            time_series_df[f"p{param_index}"] = 1
                        else:
                            time_series_df[f"p{param_index}"] = 0

                    time_series_df = time_series_df.astype(float)

                    # note: num_samples indicates the number of non-overlapping training samples
                    # note: [t : t + seq_len] -> [t + seq_len : t + 2seq_len]
                    num_samples = (len(time_series_df) - 1) // self.prediction_seq_len
                    time_series_np = time_series_df.to_numpy()

                    param_col_X = np.split(time_series_np[:num_samples * self.prediction_seq_len, :], num_samples)
                    param_col_Y = np.split(time_series_np[1:num_samples * self.prediction_seq_len + 1, :-num_cols],
                                           num_samples)

                    param_col_X, param_col_Y = np.asarray(param_col_X), np.asarray(param_col_Y)

                    param_X.append(param_col_X)
                    param_Y.append(param_col_Y)

                    # note: Currently, the entire data set are used to train, this is not the correct way
                    # note: to do testing. We are doing this for now because we do not have enough data.
                    X_train, X_test, Y_train, Y_test = train_test_split(param_col_X, param_col_Y, shuffle=True,
                                                                        test_size=0.1)
                    X_train = np.concatenate((X_train, X_test))
                    Y_train = np.concatenate((Y_train, Y_test))

                    param_quantile_timeseries.append(ParamQuantileData(X_train, X_test, Y_train, Y_test))

                self.qt_to_param_quantile_timeseries[query_template] = param_quantile_timeseries

                # note: the dictionaries map a query template to X, data to be fed into the LSTM
                # note: and Y, the ground truth for the prediction
                # note: X and Y are both lists of matrices, the length of the lists are the number of parameters
                # note: X and Y both have shape (N, seq_len, num_quantiles + n)
                # note: where N is the number of samples, seq_len is the number of data points fed into the LSTM
                # note: at a time, num_quantiles is the number of quantiles defining the distribution,
                # note: and n is the number of parameters for the parameter
                self.qt_to_param_X[query_template] = param_X
                self.qt_to_param_Y[query_template] = param_Y

    def save_checkpoint(self, ckpt_path, filename, query_template, model, epoch, optimizer=None, scheduler=None):
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

    def export_forecast_metadata(self):
        with open(qt_to_index_SAVE_PATH, "wb") as f:
            pickle.dump(self.qt_to_index, f)

        with open(qt_to_param_quantile_timeseries_SAVE_PATH, "wb") as f:
            pickle.dump(self.qt_to_param_quantile_timeseries, f)

        with open(qt_to_param_X_SAVE_PATH, "wb") as f:
            pickle.dump(self.qt_to_param_X, f)

        with open(qt_to_param_Y_SAVE_PATH, "wb") as f:
            pickle.dump(self.qt_to_param_Y, f)

        with open(qt_to_num_params_SAVE_PATH, "wb") as f:
            pickle.dump(self.query_to_num_params, f)

    def load_forecast_metadata(self):
        with open(qt_to_index_SAVE_PATH, "rb") as f:
            self.qt_to_index = pickle.load(f)

        with open(qt_to_param_quantile_timeseries_SAVE_PATH, "rb") as f:
            self.qt_to_param_quantile_timeseries = pickle.load(f)

        with open(qt_to_param_X_SAVE_PATH, "rb") as f:
            self.qt_to_param_X = pickle.load(f)

        with open(qt_to_param_Y_SAVE_PATH, "rb") as f:
            self.qt_to_param_Y = pickle.load(f)

        with open(qt_to_num_params_SAVE_PATH, "rb") as f:
            self.query_to_num_params = pickle.load(f)

    ###################################################################################################
    #########################           Model Training        #########################################
    ###################################################################################################
    def _train_epoch(self, model, X_train, Y_train, optimizer, scheduler, loss_function):
        """
        Train the LSTM for one epoch.
        @param model: LSTM model
        @param X_train: (N, seq_length, num_quantiles + n) values at time t
        @param Y_train: (N, seq_length, num_quantiles + n) values at time t + 1
        @param optimizer: optimizer
        @param scheduler: scheduler
        @param loss_function: loss_function
        @return: train loss for one epoch
        """
        model.train()

        seq = torch.tensor(X_train).to(device).float()
        labels = torch.tensor(Y_train).to(device).float()
        optimizer.zero_grad()
        prediction, _ = model(seq, None)

        assert (prediction.size() == labels.size())
        loss = loss_function(prediction, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        return loss

    def _validate(self, model, X_test, Y_test, loss_function):
        """

        @param model: LSTM model
        @param X_test: (N, seq_length, num_quantiles + n) values at time t
        @param Y_test: (N, seq_length, num_quantiles + n) values at time t + 1
        @param loss_function: loss_function
        @return: validation loss
        """
        model.eval()

        seq = torch.tensor(X_test).to(device).float()
        labels = torch.tensor(Y_test).to(device).float()

        with torch.no_grad():
            prediction, _ = model(seq, None)

        assert (prediction.size() == labels.size())
        loss = loss_function(prediction, labels)

        return loss

    def _train_model(self):
        self.qt_to_index = {}
        for template_index, (qt, param_quantile_data) in enumerate(self.qt_to_param_quantile_timeseries.items()):
            print(f"Training for Q{template_index}...")

            self.qt_to_index[qt] = template_index

            num_parameters = self.query_to_num_params[qt]
            output_size = len(self.quantiles)
            input_size = output_size + num_parameters
            model = Network(input_size, output_size, HIDDEN_SIZE, RNN_LAYERS).to(device)

            for epoch in range(EPOCHS):
                avg_train_loss = avg_val_loss = 0
                for param_index, param_quantile_data_obj in enumerate(param_quantile_data):

                    # Skip string param
                    if param_quantile_data_obj == None:
                        continue

                    X_train, X_test, Y_train, Y_test = param_quantile_data_obj.get_train_test_data()

                    # todo: can try a different loss function
                    loss_function = nn.MSELoss()

                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(X_train.shape[1] * EPOCHS))

                    avg_train_loss += self._train_epoch(model, X_train, Y_train, optimizer, scheduler, loss_function)
                    avg_val_loss += self._validate(model, X_test, Y_test, loss_function)

                avg_train_loss /= num_parameters
                avg_val_loss /= num_parameters
                print(f"epoch: {epoch + 1:3}, train_loss: {avg_train_loss:10.8f}, val_loss: {avg_val_loss:10.8f}")

            filename = f"{template_index}"
            self.save_checkpoint(MODEL_SAVE_PATH, filename, qt, model, EPOCHS)

    def fit(self, log_filename, file_type="parquet", dataframe=None, save_metadata=True):
        print("Preprocessing data...")
        self.data_preprocessor.preprocess(log_filename, file_type, dataframe)
        print("Preprocessing Done!")
        if save_metadata:
            self.data_preprocessor.save_to_file(PREPROCESSOR_SAVE_PATH)

        print("Generating training data...")
        self.generate_time_series_data()
        print("Data generation done!")

        self._train_model()

        if save_metadata:
            self.export_forecast_metadata()

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
                    seq = time_series_df.iloc[start_time: (j + 1), :].to_numpy()
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
                # (L, 1, 1 * H_out)
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

                print(f"PARAM ${i + 1} Predicted")
                fig, axes = joypy.joyplot(
                    predicted_params_df.groupby(pd.Grouper(freq="5s")),
                    hist=True,
                    bins=20,
                    overlap=0,
                    grid=True,
                    x_range=[min_val, max_val],
                )
                plt.show()

                print(f"PARAM ${i + 1} Actual")
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

    def get_parameters_for(self, query_template, timestamp, num_queries):
        target_timestamp = pd.Timestamp(timestamp)

        template_normalized_df = self.data_preprocessor.qt_to_normalized_df[query_template]
        template_dtypes = self.data_preprocessor.qt_to_dtype[query_template]
        template_stats = self.data_preprocessor.qt_to_stats[query_template]
        template_index = self.qt_to_index[query_template]
        param_X = self.qt_to_param_X[query_template]
        num_params = self.query_to_num_params[query_template]

        # todo: there is probably a better way to get the shapes
        output_size = len(self.quantiles)
        input_size = output_size + num_params

        # Get corresponding model
        model = Network(input_size, output_size, HIDDEN_SIZE, RNN_LAYERS)
        filepath = os.path.join(MODEL_SAVE_PATH, f"{template_index}")
        state_dict = torch.load(filepath)
        model.load_state_dict(state_dict["model_state"])

        generated_params = []

        for i in range(num_params):
            if template_dtypes[i] == "string":
                print("Skipping string columns")
                continue

            # Compute how many predictions need to be made
            start_timestamp = template_normalized_df.index.max()
            num_predictions = int((target_timestamp - start_timestamp) / self.prediction_interval)

            # Continuously make predictions until target_timestamp
            param_X_col = param_X[i]
            seq = param_X_col[-1][-1]
            seq = seq[None, None, :]
            seq = torch.tensor(seq).to(device).float()
            hidden = None
            for _ in tqdm(range(num_predictions)):
                # Get predicted quantiles from the model
                with torch.no_grad():
                    seq, hidden = model(seq, hidden)
                    seq = nn.functional.pad(seq, (0, num_params, 0, 0, 0, 0))
                    seq[:, :, output_size + i] = 1

            # only care about the prediction for the last step
            pred = seq.view(-1)[:-num_params]

            # Ensure prediction quantile values are strictly increasing
            pred = torch.cummax(pred, dim=0).values

            pred = pred.cpu().detach().numpy()

            # Un-normalize the quantiles
            mean, std = template_stats[i]
            if std != 0:
                pred = pred * std + mean
            else:
                pred = pred + mean

            dist = Dist(a=pred[0], b=pred[-1], name="deterministic", pred=pred, quantiles=self.quantiles)
            generated_param_ith = []

            try:
                for _ in range(num_queries):
                    generated_param_ith.append(dist.rvs())
            except:
                # If all predicted quantiles have the same value, then a continuous cdf cannot be constructed.
                # Just take any predicted quantile value as the prediction.
                for _ in range(num_queries):
                    generated_param_ith.append(pred[0])

            generated_params.append(generated_param_ith)
        return generated_params


if __name__ == "__main__":
    query_log_filename = "./preprocessed.parquet.gzip"

    forecaster = Forecaster(
        pred_interval=pd.Timedelta("2S"), pred_seq_len=3, pred_horizon=pd.Timedelta("2S"), load_metadata=False
    )
    forecaster.fit(query_log_filename)

    forecaster = Forecaster(
        pred_interval=pd.Timedelta("2S"), pred_seq_len=3, pred_horizon=pd.Timedelta("2S"), load_metadata=True
    )
    pred_result = forecaster.get_parameters_for(
        "DELETE FROM new_order WHERE NO_O_ID = $1 AND NO_D_ID = $2 AND NO_W_ID = $3",
        "2022-03-08 11:30:06.021000-0500",
        10,
    )
    print(pred_result)
