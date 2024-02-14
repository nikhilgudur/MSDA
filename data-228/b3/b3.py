from pyspark import SparkContext
from pyspark.rdd import RDD

R_FILE_PATH = "./R.txt"
S_FILE_PATH = "./S.txt"


def read_file(path: str) -> RDD:
    """read a text file from the path provided"""
    df = sc.textFile(path).map(int)

    try:
        df.collect()
    except Exception as e:
        print("File not found")
        df = sc.emptyRDD()

    return df

def mapper(key: int, val: str) -> tuple:
    return (key, val)

def reducer(a: str, b: str) -> str:
    return a + b

def filter(a: tuple) -> bool:
    return a[1] == "R"


def set_difference(r: RDD, s: RDD) -> RDD:
    """returns the set difference of r and s"""

    R_data = r.distinct().map(lambda key: mapper(key, "R"))
    S_data = s.distinct().map(lambda key: mapper(key, "S"))

    union_set = R_data.union(S_data)
    res = union_set.reduceByKey(reducer).filter(filter)
    return res.keys()



def main():
    r = read_file(R_FILE_PATH)
    s = read_file(S_FILE_PATH)

    if not r.isEmpty() and not s.isEmpty():
        r_minus_s = set_difference(r, s)
        print(tuple(r_minus_s.collect()))
    else:
        print("One or both files are empty or not found.")



if __name__ == "__main__":
    sc = SparkContext("local", "SetDifference")

    main()

    sc.stop()
