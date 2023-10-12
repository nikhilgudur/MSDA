import mysql.connector


mydb = mysql.connector.connect(
    host="data-225-g6.c6pri3k6tycv.us-east-2.rds.amazonaws.com",
    user="admin",
    password="Group6Data225",
    port="3306",
)

cursor = mydb.cursor()

# print(mydb.is_connected())
