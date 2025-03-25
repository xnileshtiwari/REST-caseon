from flask import Flask, request, jsonify
import logging
import os
from document_processing import document_chunking_and_uploading_to_vectorstore
from main_chat import start_chatting
from functools import wraps

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load valid API keys from the API_KEYS environment variable
# Expected format: a comma-separated string, e.g., "key1,key2,key3"
api_keys_str = os.environ.get("API_KEYS", "")
VALID_API_KEYS = set(filter(None, api_keys_str.split(",")))
logger.info(f"Loaded {len(VALID_API_KEYS)} API keys from environment.")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if api_key not in VALID_API_KEYS:
            logging.warning("Unauthorized access attempt.")
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_server_error(error):
    logging.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({"error": "Bad request"}), 400

# Document Processing Endpoint
@app.route("/api/v1/document/process", methods=["POST"])
@require_api_key
def process_document():
    try:
        data = request.get_json()
        if not data or "link" not in data or "unique_id" not in data:
            logging.error("Invalid request body: Missing 'link' or 'unique_id'.")
            return jsonify({
                "success": False,
                "error": 'Missing "link" or "unique_id" in request body'
            }), 400
        
        link = data["link"]
        unique_id = data["unique_id"]
        
        logging.info(f"Processing document: link={link}, unique_id={unique_id}")
        
        result = document_chunking_and_uploading_to_vectorstore(link, unique_id)
        
        logging.info(f"Document processed successfully for unique_id={unique_id}.")
        
        return jsonify({
            "success": True,
            "result": result
        }), 200

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred"
        }), 500

# Chat Endpoint
@app.route("/api/v1/chat", methods=["POST"])
@require_api_key
def chat():
    try:
        data = request.get_json()
        if not data or "index_name" not in data or "user_input" not in data:
            return jsonify({
                "success": False,
                "error": 'Missing "index_name" or "user_input" in request body'
            }), 400
        
        index_name = data["index_name"]
        user_input = data["user_input"]
        
        result = start_chatting(index_name, user_input)
        
        return jsonify({
            "success": True,
            "result": result
        }), 200

    except Exception as e:
        logging.exception("An unexpected error occurred in chat endpoint")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Health Check Endpoint
@app.route("/api/v1/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "1.0"
    }), 200

if __name__ == "__main__":
    # Get port from environment variable (Railway sets this automatically)
    port = int(os.environ.get("PORT", 5000))
    
    if os.environ.get("ENVIRONMENT") == "production":
        # Production: use waitress
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)
    else:
        # Development: use Flask's built-in server
        app.run(debug=False, host="0.0.0.0", port=port)
