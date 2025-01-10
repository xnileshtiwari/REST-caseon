import pymysql
from datetime import datetime
from typing import List, Dict, Tuple
from connection import getconnection

connection = getconnection()



def get_trending_on_date(target_date: str) -> Tuple[bool, str, List[Dict]]:
    """
    Get trending analysis for a specific date.
    
    Args:
        connection: PyMySQL database connection object
        target_date: str, date in format 'YYYY-MM-DD'
    
    Returns:
        tuple: (success: bool, message: str, trending_data: List[Dict])
            trending_data contains dicts with {'category': str, 'count': int}
    """
    cursor = None
    try:
        # Validate date format
        try:
            parsed_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        except ValueError:
            return False, "Invalid date format. Please use YYYY-MM-DD", []

        cursor = connection.cursor()
        
        # First, get all column names except 'date'
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'cat_is_trending' 
            AND TABLE_SCHEMA = 'defaultdb'
            AND COLUMN_NAME != 'date'
        """)
        
        columns = [row['COLUMN_NAME'] for row in cursor.fetchall()]
        
        if not columns:
            return False, "No trending categories found in the table", []
            
        # Build dynamic SQL query to get values for all columns on target date
        columns_str = ', '.join(columns)
        query = f"""
        SELECT {columns_str}
        FROM cat_is_trending
        WHERE date = %s
        """
        
        cursor.execute(query, (target_date,))
        result = cursor.fetchone()
        
        if not result:
            return False, f"No data found for date {target_date}", []
            
        # Convert to list of dictionaries with non-zero values
        trending_list = [
            {'category': col, 'count': result[col]}
            for col in columns
            if result[col] > 0  # Exclude zero values
        ]
        
        # Sort by count in descending order
        trending_list.sort(key=lambda x: x['count'], reverse=True)
        
        if not trending_list:
            return True, f"No trending items found for date {target_date} (all values were 0)", []
            
        return True, "Data retrieved successfully", trending_list

    except pymysql.MySQLError as e:
        return False, f"Database error occurred: {str(e)}", []
        
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}", []
        
    finally:
        if cursor:
            cursor.close()

# Helper function to format the trending results nicely
def print_trending_results(success: bool, message: str, trending_data: List[Dict]) -> None:
    """
    Pretty print the trending results.
    """
    print(f"Status: {'Success' if success else 'Failed'}")
    print(f"Message: {message}")
    
    if trending_data:
        print("\nTrending Categories:")
        print("-" * 40)
        print(f"{'Category':<20} {'Count':>10}")
        print("-" * 40)
        for item in trending_data:
            print(f"{item['category']:<20} {item['count']:>10}")
    print()

# Example usage
def test_trending_analysis():
    # Test with a valid date
    target_date = "2024-12-22"  # Replace with your target date
    success, message, trending_data = get_trending_on_date(connection, target_date)
    print_trending_results(success, message, trending_data)
    

# Run the test if needed
test_trending_analysis()