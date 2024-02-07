from pyspark.rdd import RDD
from pyspark.context import SparkContext

FILE_PATH = "./input.txt"
SEV = ["INFO", "DEBUG", "WARN", "ERROR", "CRITICAL"]

def main():
    df = read_file(FILE_PATH)

    if not df.isEmpty():
        df = count_severity(df)
        print(df.collect())
    else:
        print("File is empty")


def read_file(path: str) -> RDD:
    """read a text file from the path provided"""


    df = sc.textFile(path)

    try:
        df.collect()
    except Exception as e:
        print("File not found")
        df = sc.emptyRDD()

    return df


def count_severity(df: RDD) -> RDD:
    """counts the number of occurrences of each severity in the file"""

    # map each line to a tuple of (severity, 1).
    # This will return None for lines that do not meet the condition in the count function
    df = df.map(count)

    # remove None values from df
    df = df.filter(lambda x: x is not None)

    df = df.reduceByKey(lambda a, b: a + b)

    return df


def count(x: str) -> tuple:
    if x.split(" ")[0] in SEV:
        return (x.split(" ")[0], 1)



if __name__ == "__main__":
    sc = SparkContext().getOrCreate()

    main()

    sc.stop()