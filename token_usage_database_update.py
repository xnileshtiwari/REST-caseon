import pymysql
from datetime import datetime
from connection import getconnection

def update_token_usage(input_tokens: int, output_tokens: int):
    """
    Update or insert token usage for the current date.
    
    Args:
        input_tokens (int): Number of input tokens to add.
        output_tokens (int): Number of output tokens to add.
    """
    # Input validation
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Input and output tokens must be non-negative.")

    connection = getconnection()
    if connection is None:
        return

    try:
        cursor = connection.cursor()

        # SQL query to increment or insert token values
        query = """
        INSERT INTO token_usage (Date, Input_token, Output_token)
        VALUES (CURDATE(), %s, %s)
        ON DUPLICATE KEY UPDATE
            Input_token = Input_token + VALUES(Input_token),
            Output_token = Output_token + VALUES(Output_token);
        """

        # Execute the query
        cursor.execute(query, (input_tokens, output_tokens))
        
        # Commit the transaction
        connection.commit()
        print(f"Token usage updated successfully for {datetime.now().date()}. || Input tokens: {input_tokens}, Output tokens: {output_tokens}")

    except pymysql.MySQLError as e:
        # Handle database errors
        print(f"Database error: {e}")
        connection.rollback()

    finally:
        # Clean up the connection
        cursor.close()
        connection.close()

