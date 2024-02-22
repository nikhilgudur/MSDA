from pyspark import SparkContext
from pyspark.rdd import RDD


A_MATRIX = "./A.dat"
B_MATRIX = "./B.dat"


def read_file_mapper(line: str) -> str:

    if len(line) == 0 or line.startswith("#"):
        return None

    line = line.split(",")
    line = list(map(lambda x: int(x), line))
    return list(line)


def read_file(path: str) -> RDD:
    """read a text file from the path provided"""
    df = sc.textFile(path).map(read_file_mapper).filter(
        lambda x: x is not None)

    try:
        df.collect()
    except Exception as e:
        print("File not found")
        df = sc.emptyRDD()

    return df


def first_mapper(row: list, i: int, matrix: str) -> list:
    mapping = []
    for j in range(len(row)):
        if matrix == "B":
            mapping.append((i, (matrix, j, row[j])))
        else:
            mapping.append((j, (matrix, i, row[j])))
    return mapping


def matrix_multiplication(A: RDD, B: RDD) -> RDD:
    A_data = A.zipWithIndex().flatMap(
        lambda row: first_mapper(row[0], row[1], "A"))
    B_data = B.zipWithIndex().flatMap(
        lambda row: first_mapper(row[0], row[1], "B"))

    joined_matrix = A_data.union(B_data)

    intermediate_result = joined_matrix.groupByKey().flatMapValues(lambda x: [
        ((A[1], B[1]), int(A[2]) * int(B[2]))
        for A in x if A[0] == 'A'
        for B in x if B[0] == 'B'
    ])

    output = intermediate_result.map(lambda x: (
        x[1][0], x[1][1])).reduceByKey(lambda x, y: x + y).map(lambda x: x[1])

    return output


def main():
    A = read_file(A_MATRIX)
    B = read_file(B_MATRIX)

    temp = A.collect()

    if len(temp[0]) != len(B.collect()):
        print("Matrix multiplication is not possible.")
        return

    result = matrix_multiplication(A, B)

    print(result.collect())


if __name__ == "__main__":
    sc = SparkContext("local", "MatrixMultiplication")

    main()

    sc.stop()
