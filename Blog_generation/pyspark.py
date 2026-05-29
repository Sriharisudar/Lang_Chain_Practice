import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import format_number, col
# Create a Spark session
spark = SparkSession.builder \
    .appName("test") \
    .config("spark.jars.packages") \
    .getOrCreate()
df = spark.createDataFrame([("000010000",)], ["num"])
df2 = df.withColumn("decimal_value", format_number(col("num").cast("int"),4))
df2.show()

spark = SparkSession.builder.appName("test").getOrCreate()