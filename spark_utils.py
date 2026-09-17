"""Shared helpers: config loading and Spark session creation."""

import yaml
from pyspark.sql import SparkSession


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_spark_session(config):
    """Create (or fetch) a local SparkSession using the settings in config.yaml.

    Running with master="local[*]" uses all CPU cores on this machine as
    Spark "executors" — the same DataFrame/MLlib API works unchanged on a
    real multi-node cluster by pointing `spark_master` at one instead.
    """
    spark = (
        SparkSession.builder
        .appName(config.get("spark_app_name", "pyspark-mllib-pipeline"))
        .master(config.get("spark_master", "local[*]"))
        .config("spark.sql.shuffle.partitions", config.get("spark_shuffle_partitions", 8))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark