FROM ghcr.io/mlflow/mlflow:v2.17.2

# The upstream image has no PostgreSQL driver.
RUN pip install --no-cache-dir psycopg2-binary==2.9.10

EXPOSE 5000
