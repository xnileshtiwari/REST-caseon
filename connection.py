import pymysql
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

def getconnection():
    try:
        timeout = 10
        connection = pymysql.connect(
            charset="utf8mb4",
            connect_timeout=timeout,
            cursorclass=pymysql.cursors.DictCursor,
            db="defaultdb",
            host=os.getenv("MYSQL_HOST"),
            password=os.getenv("MYSQL_PASSWORD"),
            read_timeout=timeout,
            port=10849,
            user=os.getenv("MYSQL_USER"),
            write_timeout=timeout,
        )
        return connection
    except Exception as e:
        print(f"An error occurred in database connection: {str(e)}")
        return None



