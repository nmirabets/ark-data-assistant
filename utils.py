import mysql.connector
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Function to execute MySQL query and return the result as a DataFrame
def execute_query(query):
    try:
        db_config = {
            "host": "ark-digest.crubyqmjsrku.eu-west-3.rds.amazonaws.com",
            "user": os.getenv('DB_USER'),
            "password": os.getenv('DB_PASS'),
            "database": "ark_digest",
        }

        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        cursor.execute(query)
        result = cursor.fetchall()

        if result:
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(result, columns=columns)
            return df
        else:
            return None

    except mysql.connector.Error as e:
        return f"Error: {e}"

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()