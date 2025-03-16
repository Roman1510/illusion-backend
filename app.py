from flask import Flask
from flask_cors import CORS
from config import logger, TEMP_DIR
from routes import init_routes

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# Initialize routes
init_routes(app)

if __name__ == '__main__':
    logger.info(f"Starting Flask app with temp directory: {TEMP_DIR}")
    app.run(debug=True, host='0.0.0.0', port=5000)