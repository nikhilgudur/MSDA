from pyspark.sql import SparkSession

from pyspark.sql.functions import *
from pyspark.sql.types import *
# Streaming via socket
spark = SparkSession.builder.appName("streamCsv").getOrCreate()
# sc = SparkContext().getOrCreate()


rawdata = spark.readStream.format("socket").option(
    "host", "localhost").option("port", 9999).option(
    "includeTimestamp", True).load()
# rawdata = spark.read.csv("./test.txt", header=True, inferSchema=True)
query = rawdata.select((rawdata.value).alias("product"), (rawdata.timestamp).alias(
    "time")).groupBy(window("time", "1minutes"), "product").count().sort(desc("window"))
result = query.writeStream.format("console").outputMode(
    "complete").start().awaitTermination()
result.stop()
