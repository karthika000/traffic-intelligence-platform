#!/bin/sh
# Runs once, on first start with an empty postgres_data volume.
# MLflow gets its own database so its tables stay out of the application schema.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
CREATE DATABASE "$MLFLOW_DB" OWNER "$POSTGRES_USER";
EOSQL
