import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext


def main():

    sc = SparkContext(appName="b10")
    ssc = StreamingContext(sc, 1)

    hostname = ""
    port = 0

    if len(sys.argv) < 3:
        print("Usage: b10.py <hostname> <port>")
        sys.exit(-1)

    else:
        hostname = sys.argv[1]
        port = int(sys.argv[2])

    try:
        lines = ssc.socketTextStream(hostname, port)
        numbers = lines.flatMap(lambda line: line.split(
            " ")).flatMap(lambda line: line.split(",")).map(lambda number: round(float(number)))
        numberCounts = numbers.map(lambda number: (
            number, 1)).reduceByKey(lambda a, b: a+b)
        numberCounts.pprint()

        total = numbers.reduce(lambda a, b: a+b)

        total.pprint()
        ssc.start()
        ssc.awaitTermination()
    except:
        ssc.stop()
        sc.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
