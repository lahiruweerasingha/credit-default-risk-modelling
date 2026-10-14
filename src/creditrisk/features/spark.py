"""Spark session factory.

`SPARK_MASTER` selects the cluster: `spark://spark-master:7077` for the standalone cluster in
docker-compose, `local[*]` for tests and CI.
"""

from __future__ import annotations

import os
import socket

from pyspark.sql import SparkSession


def get_spark(app_name: str = "creditrisk") -> SparkSession:
    master = os.environ.get("SPARK_MASTER", "local[*]")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.ansi.enabled", "false")  # invalid casts -> NULL, handled explicitly
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "64"))
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "2g"))
        .config("spark.executor.memory", os.environ.get("SPARK_EXECUTOR_MEMORY", "3g"))
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    if master.startswith("spark://"):
        # Executors on the worker container must reach the driver container by IP.
        builder = builder.config("spark.driver.host", socket.gethostbyname(socket.gethostname()))
        builder = builder.config("spark.driver.bindAddress", "0.0.0.0")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
