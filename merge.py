from constants import CLUSTER_ASSIGNMENT_CSV, QUERY_TIMESERIES_CSV, CLUSTER_TIMESERIES_CSV
import pandas as pd

cluster_assignment = pd.read_csv(CLUSTER_ASSIGNMENT_CSV, usecols=[
                                 "cluster", "query_template"])
query_timeseries = pd.read_csv(QUERY_TIMESERIES_CSV, parse_dates=[
                               "log_time"], usecols=['count', "query_template", "log_time"])

merged = query_timeseries.merge(cluster_assignment, on="query_template")
cluster_timeseries = merged.groupby(["cluster", "log_time"])["count"].sum().reset_index()
cluster_timeseries.to_csv(CLUSTER_TIMESERIES_CSV, index=False)