import requests
import os
from urllib.parse import urlparse

def download_pdf(url, output_path=None):
    """
    Download a PDF file from a URL with proper error handling.
    
    Args:
        url (str): The URL of the PDF file
        output_path (str, optional): The path where the PDF should be saved
                                   If None, derives filename from URL
    
    Returns:
        str: Path to the downloaded file if successful
    """
    try:
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        
        # Send GET request with headers
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Verify content type is PDF
        content_type = response.headers.get('content-type', '').lower()
        
        # Set fixed output path to 'sample.pdf'
        if output_path is None:
            output_path = 'sample.pdf'
        
        # Save the PDF file
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        return output_path
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        if response.status_code == 406:
            print("406 Error: Server doesn't accept the request. Try adjusting headers.")
        raise
    except requests.exceptions.ConnectionError:
        print("Connection error occurred. Please check your internet connection.")
        raise
    except requests.exceptions.Timeout:
        print("Timeout error occurred. The server took too long to respond.")
        raise
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the PDF: {e}")
        raise
    except IOError as e:
        print(f"An error occurred while saving the file: {e}")
        raise

