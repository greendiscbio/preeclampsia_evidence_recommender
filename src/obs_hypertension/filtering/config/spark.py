from pyspark.sql import SparkSession
APP_NAME = "pregnancy_hypertension"

SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "4g"

SPARK_CONF = {
    "spark.sql.debug.maxToStringFields": "10000",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
}

def init_spark(app_name: str):
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[4]")  # ⚠️ NO local[*]
        .config("spark.executor.instances", "1")
        .config("spark.executor.cores", "4")
        .config("spark.task.cpus", "1")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "32g")
        .config("spark.executor.memory", "32g")
        .getOrCreate()
    )

    return spark