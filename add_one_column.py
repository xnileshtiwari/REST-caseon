import pymysql
from connection import getconnection

connection = getconnection()


def add_one_to_column(column_name):
    """
    Adds a new column to the cat_is_trending table.
    
    Args:
        connection: PyMySQL database connection object
        column_name: str, name of the column to add
        data_type: str, SQL data type for the new column (default: "INT")
    
    Returns:
        tuple: (bool, str) - (Success status, Message)
        
    Raises:
        pymysql.MySQLError: If there's any database error
    """
    cursor = None
    data_type="INT"
    try:
        # Sanitize column name to prevent SQL injection
        # Only allow alphanumeric characters and underscores
        if not column_name.replace('_', '').isalnum():
            return False, "Column name can only contain letters, numbers, and underscores"
            
        cursor = connection.cursor()
        
        # Add the new column
        alter_query = f"""
        ALTER TABLE cat_is_trending 
        ADD COLUMN {column_name} {data_type} NOT NULL DEFAULT 0
        """
        
        cursor.execute(alter_query)
        connection.commit()
        return True, f"Successfully added column '{column_name}'"
        
    except pymysql.err.OperationalError as e:
        # Handle the specific case where column already exists
        if e.args[0] == 1060:  # MySQL error code for duplicate column
            return False, f"Column '{column_name}' already exists in the table"
        return False, f"Database operational error: {str(e)}"
        
    except pymysql.MySQLError as e:
        return False, f"Database error occurred: {str(e)}"
        
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}"
        
