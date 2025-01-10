import pymysql
from datetime import datetime
from connection import getconnection

connection = getconnection()

def calculate_token_cost(date: str, input_token_cost: float, output_token_cost: float) -> dict:
    """
    Calculate the total costs of input and output tokens for a given date.

    Args:
        date (str): The date in 'YYYY-MM-DD' format for which the costs are to be calculated.
        input_token_cost (float): Cost of one input token in dollars.
        output_token_cost (float): Cost of one output token in dollars.

    Returns:
        dict: A dictionary containing total input cost, total output cost, and combined total cost.
    """
    try:
        # Validate the date format
        try:
            valid_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Date must be in 'YYYY-MM-DD' format.")

        # Validate token costs
        if input_token_cost < 0 or output_token_cost < 0:
            raise ValueError("Token costs must be non-negative.")

        cursor = connection.cursor()

        # Fetch the token usage for the given date
        query = '''
        SELECT Input_token, Output_token
        FROM token_usage
        WHERE Date = %s;
        '''
        cursor.execute(query, (valid_date,))
        result = cursor.fetchone()

        # Handle case where no data is found for the given date
        if not result:
            raise ValueError(f"No token usage data found for date: {valid_date}")

        input_tokens = result["Input_token"]
        output_tokens = result["Output_token"]

        # Calculate costs
        total_input_cost = input_tokens * input_token_cost
        total_output_cost = output_tokens * output_token_cost
        total_cost = total_input_cost + total_output_cost

        return {
            "date": str(valid_date),
            "Total input token cost $": round(total_input_cost, 2),
            "Total output token cost $": round(total_output_cost, 2),
            "Total token cost $": round(total_cost, 2),
        }

    except pymysql.MySQLError as e:
        # Handle database errors
        raise RuntimeError(f"Database error: {e}")


