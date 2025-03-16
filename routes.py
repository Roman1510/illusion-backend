from flask import request, jsonify
from image_processor import ImageProcessor
import cv2
from config import FRONT_NO_BG_PATH, SIDE_NO_BG_PATH, MERGED_IMAGE_PATH
import logging

logger = logging.getLogger(__name__)

def init_routes(app):
    """Initialize all routes for the Flask app."""
    
    @app.route('/upload', methods=['POST'])
    def upload_images():
        """
        Handle image uploads, remove backgrounds, detect landmarks, merge images, and return the result.
        
        Expects two images in the request:
            - 'front': The first image
            - 'side': The second image
        
        Returns:
            JSON response with:
                - mergedImage: Base64-encoded merged image
                - landmarks: Array of two objects with detected coordinates for each image
        """
        if 'front' not in request.files or 'side' not in request.files:
            return jsonify({'error': 'Missing files, both front and side images are required'}), 400

        processor = ImageProcessor()

        try:
            # Decode images
            logger.info("Decoding uploaded images")
            front_file = request.files['front']
            side_file = request.files['side']
            front_img = processor.decode_image(front_file)
            side_img = processor.decode_image(side_file)

            # Resize images for consistency
            front_img = cv2.resize(front_img, (300, 300))
            side_img = cv2.resize(side_img, (300, 300))

            # Remove backgrounds
            logger.info("Removing backgrounds from images")
            front_no_bg = processor.remove_background(front_img)
            side_no_bg = processor.remove_background(side_img)

            # Save processed images
            cv2.imwrite(FRONT_NO_BG_PATH, front_no_bg)
            cv2.imwrite(SIDE_NO_BG_PATH, side_no_bg)

            # Detect landmarks
            logger.info("Detecting landmarks for both images")
            front_landmarks = processor.detect_landmarks(front_no_bg, 300, 300, is_side_profile=False)
            side_landmarks = processor.detect_landmarks(side_no_bg, 300, 300, is_side_profile=True)

            # Merge images
            logger.info("Merging images")
            merged_image = processor.merge_images(front_no_bg, side_no_bg, front_landmarks, side_landmarks)

            # Convert to grayscale
            if merged_image.shape[2] == 4:
                merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGRA2GRAY)
                merged_image = cv2.cvtColor(merged_image, cv2.COLOR_GRAY2BGRA)

            cv2.imwrite(MERGED_IMAGE_PATH, merged_image)

            # Encode merged image to base64
            merged_base64 = processor.encode_image_to_base64(merged_image)

            # Prepare response
            response = {
                'mergedImage': merged_base64,
                'landmarks': [
                    {'image': 'front', 'coordinates': front_landmarks},
                    {'image': 'side', 'coordinates': side_landmarks}
                ]
            }
            logger.info("Successfully processed and merged images")
            return jsonify(response)

        except Exception as e:
            logger.error(f"Error in /upload endpoint: {str(e)}")
            return jsonify({'error': f'Failed to process images: {str(e)}'}), 500

    @app.route('/adjust', methods=['POST'])
    def adjust_images():
        """
        Placeholder endpoint for adjusting images based on points and coordinates.
        
        Returns:
            JSON response confirming the endpoint is accessible
        """
        return jsonify({
            'status': 'Adjust endpoint is working',
            'message': 'This is a placeholder for future adjustments'
        })