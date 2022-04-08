DEBUG_MYSQL_LOG = "./admission/magneto.*"
# DEBUG_MYSQL_CSV = "/Users/jackiedong/Documents/CMU/2022-spring/15799/admission_data/magneto.log.2017-07-02-anon.csv"
DEBUG_POSTGRESQL_CSV = "./admission/mysql_to_postgres_raw.csv"
DEBUG_POSTGRESQL_CSV_PARSED = "./parsed_admission/postgres_parsed.csv"
CLUSTER_ASSIGNMENT_CSV = "./parsed_admission/cluster_assignment.csv"
QUERY_TIMESERIES_CSV = "./parsed_admission/query_timeseries.csv"
CLUSTER_TIMESERIES_CSV = "./parsed_admission/cluster_timeseries.csv"

PG_LOG_DTYPES = {
    "log_time": str,
    "user_name": str,
    "database_name": str,
    "process_id": "Int64",
    "connection_from": str,
    "session_id": str,
    "session_line_num": "Int64",
    "command_tag": str,
    "session_start_time": str,
    "virtual_transaction_id": str,
    "transaction_id": "Int64",
    "error_severity": str,
    "sql_state_code": str,
    "message": str,
    "detail": str,
    "hint": str,
    "internal_query": str,
    "internal_query_pos": "Int64",
    "context": str,
    "query": str,
    "query_pos": "Int64",
    "location": str,
    "application_name": str,
    "backend_type": str,
    "leader_pid": "Int64",
    "query_id": "Int64",
}

