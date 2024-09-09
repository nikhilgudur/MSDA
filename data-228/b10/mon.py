import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext(appName="bonus10")
ssc = StreamingContext(sc, 1)


def main():

    hostname = ""
    port = 0

    if len(sys.argv) < 3:
        print("enter port number and hostname")
        sys.exit(-1)

    else:
        hostname = sys.argv[1]
        port = int(sys.argv[2])

    lines = ssc.socketTextStream(hostname=hostname, port=port)
    numbers = lines.flatMap(lambda line: line.split(
        " "))
    cleanedNumbers = numbers.flatMap(lambda line: line.split(
        ",")).map(lambda number: round(float(number)))
    count = cleanedNumbers.map(lambda number: (
        number, 1)).reduceByKey(lambda a, b: a+b)
    count.pprint()

    totalCount = cleanedNumbers.reduce(lambda a, b: a+b)

    totalCount.pprint()

    ssc.start()
    ssc.awaitTermination()


main()
