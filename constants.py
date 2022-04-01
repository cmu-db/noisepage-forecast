DEBUG_MYSQL_LOG = "/home/cmu-db-nuc-wan/admissions/magneto.log.2016-09-04"
DEBUG_MYSQL_CSV = "/tmp/meowmeow.csv"
DEBUG_POSTGRESQL_CSV = "/tmp/meowmeow2.csv"

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
