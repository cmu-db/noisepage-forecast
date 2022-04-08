import csv
import re

import dask
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from constants import (
    DEBUG_MYSQL_LOG,
    DEBUG_POSTGRESQL_CSV,
    PG_LOG_DTYPES,
)

dask.config.set(scheduler="synchronous")  # overwrite default with multiprocessing scheduler


def convert_mysql_general_log_to_mysql_csv(mysql_log_path, output_csv_path):
    """
    Convert the general log into CSV format.

    By default, the MySQL general log format is not set up for read_csv.
    There is a general lack of clear delineation between different queries,
    even vim chokes trying to open some of the log files.
    This converts the annoying input into slightly cleaner CSV,
    with headers

    TODO(WAN): Write a wrapper around IOBase to avoid writing an intermediate file.

    Parameters
    ----------
    mysql_log_path : Path
        The MySQL general log to be converted.
    output_csv_path : Path
        The output destination for the CSVfied MySQL query log.
    """

    # Regexes for recognizing various MySQL query log constructs.
    regex_header = re.compile(r"[\s\S]*Time\s+Id\s+Command\s+Argument")
    regex_date_id = re.compile(r"^(\d+.*)Z(\d+)")
    regex_time_id_command_argument = re.compile(r"(\d+.*)Z(\d+) (Connect|Init DB|Query|Quit|Statistics)\t([\s\S]*)")

    with open(output_csv_path, "w", encoding="utf-8") as output_csv:
        writer = csv.writer(output_csv, lineterminator="\n", quoting=csv.QUOTE_ALL)

        def buffer_to_line(buffer):
            # Join the buffer.
            joined_buf = "\n".join(buffer)
            # PostgreSQL vs MySQL things.
            joined_buf = joined_buf.replace("'", "'")
            # TODO(WAN): Normalize the query? But valid queries not guaranteed, e.g., log-raw.
            return joined_buf

        def write_line(line):
            # Parse the line into a (time, id, command, argument)
            match = regex_time_id_command_argument.match(line)
            if match is None:
                assert regex_header.match(line) is not None, f"Bad line: {line}"
                # If control flow reaches this point, the line is hopefully junk.
            else:
                writer.writerow(match.groups())

        writer.writerow(["Time", "Id", "Command", "Argument"])
        buffer = []
        # num_lines is wasteful, but eh, progress tracking is nice.
        with open(mysql_log_path, "r", encoding="latin-1") as dummy_file:
            num_lines = sum(1 for _ in dummy_file)
        with tqdm(open(mysql_log_path, "r", encoding="latin-1"), total=num_lines) as mysql_log:
            # Iterate over each line in the query log as delimited by \n.
            # Note that this is not a complete log entry,
            # because query strings can contain \n's as well.
            for line in mysql_log:
                # First, remove any trailing \n's.
                line = line.rstrip("\n")
                # If there is no date, this is _probably_ part of the previous line.
                # TODO(WAN): Except for when it isn't.
                if regex_date_id.match(line) is None:
                    # Continuation of previous line.
                    buffer.append(line)
                    continue
                # Otherwise, finish the current line and initialize the next.
                write_line(buffer_to_line(buffer))
                buffer = [line]
        write_line(buffer_to_line(buffer))


def convert_mysql_csv_to_postgresql_csv(mysql_csv_path, output_csv_path):
    """
    Convert a CSVfied MySQL query log to a PostgreSQL csvlog.

    Parameters
    ----------
    mysql_log_path : Path
        The CSVfied MySQL general log to be converted.
    output_csv_path : Path
        The output destination for the PostgreSQL query log.
    """

    # blocksize=None is necessary. dask defaults to chunking at \n boundaries,
    # but since our queries can contain \n tokens, we can't let it do that and
    # must live without the parallelism.
    mysql_df = dd.read_csv(mysql_csv_path, blocksize=None, names=["date", "time", "Id", "Command", "Argument"])

    def augment(df):
        thread_id = df["Id"].iloc[0]
        df["Time"] = df["date"] + " " + df["time"]

        # TODO(WAN): Right now, we assume autocommit=1. But maybe we can parse this out.
        df["session_id"] = thread_id
        df["session_line_num"] = range(df.shape[0])
        df["virtual_transaction_id"] = [f"AAC/{thread_id}/{n}" for n in range(df.shape[0])]
        df = df.drop(columns=["Id", "date", "time"])

        # TODO(WAN): This is kind of an abuse of PostgreSQL portal names.
        df["message"] = "execute " + df["Command"] + ": " + df["Argument"]
        df = df.drop(columns=["Command", "Argument"])
        df = df.rename(columns={"Time": "log_time"})

        for key in PG_LOG_DTYPES:
            if key not in df.columns:
                df[key] = pd.NA
        return df[PG_LOG_DTYPES.keys()]

    postgresql_df = mysql_df.groupby("Id").apply(augment, meta=PG_LOG_DTYPES)
    postgresql_df = postgresql_df.sort_values("log_time")
    postgresql_df.to_csv(
        output_csv_path, single_file=True, index=False, header=False, quoting=csv.QUOTE_ALL,
    )


def main():
    pbar = ProgressBar()
    pbar.register()
    # convert_mysql_general_log_to_mysql_csv(DEBUG_MYSQL_LOG, DEBUG_MYSQL_CSV)
    convert_mysql_csv_to_postgresql_csv(DEBUG_MYSQL_LOG, DEBUG_POSTGRESQL_CSV)


if __name__ == "__main__":
    main()
