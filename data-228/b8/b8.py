import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame


FILE_PATH = "./input.dat"


def read_data(file_path):
    df = sc.textFile(file_path).flatMap(lambda line: line.split("\n"))

    try:
        df.collect()
    except Exception as e:
        print(e)
        sys.exit(1)

    return df


def create_graph(df):

    df = df.map(lambda x: x.split(" ")).filter(lambda x: x[0] != "")

    person1 = df.map(lambda x: (x[0],))
    person2 = df.map(lambda x: (x[2],))

    peopleVertices = person1.union(person2).distinct().toDF(["id"])

    edges = df.map(lambda x: (x[0], x[2], x[4], x[6])).toDF(
        ["src", "dst", "relationship", "year"])

    graph = GraphFrame(peopleVertices, edges)

    return graph


def find_friends_of_friends(person, graph):

    friends_of_friends = graph.find("(a)-[e1]->(b); (b)-[e2]->(c)").filter(
        F.col("a.id") == person).select("c.id", "e2.year").distinct()

    return friends_of_friends.orderBy(F.asc("year"))


def main():

    df = read_data(FILE_PATH)

    graph = create_graph(df)

    for person in graph.vertices.rdd.map(lambda row: row.id).collect():
        print(person)
        new_friends = find_friends_of_friends(person, graph)
        if not new_friends.rdd.isEmpty():
            print(f"Introducing new friends to {person}:")
            new_friends.show()


if __name__ == "__main__":
    spark = SparkSession.builder.appName("b8").config(
        "spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12").getOrCreate()
    sc = spark.sparkContext

    main()

    sc.stop()
