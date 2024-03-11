import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import col, isnan, when, count, concat, sort_array, array, expr
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline


POKER_HANDS = "./poker-hand-training-true.data"
POKER_HANDS_TEST = "./poker-hand-testing.data"


def read_file(file_path: str):
    """read a text file from the path provided"""

    schema = StructType([
        StructField("Suit1", IntegerType(), True),
        StructField("Rank1", IntegerType(), True),
        StructField("Suit2", IntegerType(), True),
        StructField("Rank2", IntegerType(), True),
        StructField("Suit3", IntegerType(), True),
        StructField("Rank3", IntegerType(), True),
        StructField("Suit4", IntegerType(), True),
        StructField("Rank4", IntegerType(), True),
        StructField("Suit5", IntegerType(), True),
        StructField("Rank5", IntegerType(), True),
        StructField("Class", IntegerType(), True)
    ])

    df = spark.read.csv(file_path, schema=schema, header=False)

    try:
        df.collect()
        return df
    except Exception as e:
        print("File not found")
        sys.exit(1)


def get_missing_values(df):
    """Find the missing values in the dataframe"""

    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
              ).show()


def get_correlation(df):
    """Find the correlation between the columns in the dataframe"""

    for col1 in df.columns:
        for col2 in df.columns:
            if col1 is not col2:
                print(
                    f"Correlation between {col1} and {col2}: {df.stat.corr(col1, col2)}")


def find_duplicates(df):
    """Find the duplicate rows in the dataframe and remove them"""

    df_combined = df.withColumn("Set1", concat(df["Suit1"], df["Rank1"])) \
        .withColumn("Set2", concat(df["Suit2"], df["Rank2"])) \
        .withColumn("Set3", concat(df["Suit3"], df["Rank3"])) \
        .withColumn("Set4", concat(df["Suit4"], df["Rank4"])) \
        .withColumn("Set5", concat(df["Suit5"], df["Rank5"]))

    df_combined = df_combined.drop("Suit1", "Rank1", "Suit2", "Rank2",
                                   "Suit3", "Rank3", "Suit4", "Rank4", "Suit5", "Rank5")

    for i in range(1, 6):
        df_combined = df_combined.withColumn(
            f"Set{i}", col(f"Set{i}").cast(IntegerType()))

    df_combined = df_combined.withColumn("SortedSets", sort_array(
        array([col("Set1"), col("Set2"), col("Set3"), col("Set4"), col("Set5")])))

    for i in range(1, 6):
        df_combined = df_combined.withColumn(
            f"Set{i}", col("SortedSets").getItem(i-1))

    df_combined = df_combined.drop("SortedSets")

    num_duplicates = df.count() - df.dropDuplicates(df.columns).count()

    print("Number of duplicate rows:", num_duplicates)

    df_unique = df_combined.dropDuplicates()

    return df_unique


def modify_columns(df):
    """Modify the columns in the dataframe"""

    for i in range(1, 6):

        df = df.withColumn(f"Suit{i}", expr(
            f"substring(cast(Set{i} as string), 1, 1)").cast("integer"))
        df = df.withColumn(f"Rank{i}", expr(
            f"substring(cast(Set{i} as string), 2, length(cast(Set{i} as string))-1)").cast("integer"))

    for i in range(1, 6):
        df = df.drop(f"Set{i}")

    return df


def train_model(df, df_test):

    assembler = VectorAssembler(
        inputCols=["Suit1", "Rank1", "Suit2", "Rank2", "Suit3",
                   "Rank3", "Suit4", "Rank4", "Suit5", "Rank5"],
        outputCol="features")

    indexer = StringIndexer(inputCol="Class", outputCol="label")

    rf = RandomForestClassifier(
        labelCol="label", featuresCol="features", numTrees=10)

    pipeline = Pipeline(stages=[assembler, indexer, rf])

    model = pipeline.fit(df)

    predictions = model.transform(df_test)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    print("Accuracy = %g" % accuracy)

    print("Test Error = %g" % (1.0 - accuracy))

    return model


def main():
    df = read_file(POKER_HANDS)
    df_test = read_file(POKER_HANDS_TEST)

    # To find the missing values
    get_missing_values(df)

    df.describe().show()

    # To drop missing values
    df = df.na.drop()

    # To find the missing values
    get_missing_values(df)

    # To find the correlation between the columns
    get_correlation(df)

    unique_df = find_duplicates(df)

    unique_df.show()

    modified_df = modify_columns(unique_df)
    modified_df.show()

    model = train_model(modified_df, df_test)

    model.write().overwrite().save("/Users/nikhilgudur/msda/data-228/b6/model")


if __name__ == "__main__":
    spark = SparkSession.builder.appName("b5").getOrCreate()

    main()

    spark.stop()
