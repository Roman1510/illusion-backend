import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory to store temporary images
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# File paths for temporary storage
FRONT_NO_BG_PATH = os.path.join(TEMP_DIR, "front_no_bg.png")
SIDE_NO_BG_PATH = os.path.join(TEMP_DIR, "side_no_bg.png")
MERGED_IMAGE_PATH = os.path.join(TEMP_DIR, "merged_image.png")

if __name__ == "__main__":
    logger.info("Configuration loaded successfully")