import logging
from flask import Flask
from service import Service
from routes import register_routes

# Patch for transformers to avoid torch.fx import error in environments without PyTorch FX support
import transformers.utils.import_utils as _import_utils
if not hasattr(_import_utils, "is_torch_fx_available"):
    _import_utils.is_torch_fx_available = lambda: False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize service
service = Service("127.0.0.1")

# Register routes
register_routes(app, service)


def main():
    """Main function - run as Flask API server"""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
