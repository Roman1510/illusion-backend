from flask import request, jsonify # type: ignore
from image_processor import ImageProcessor
import cv2 # type: ignore
import numpy as np # type: ignore
import os
import base64
import logging
from config import FRONT_NO_BG_PATH, SIDE_NO_BG_PATH, MERGED_IMAGE_PATH, logger

def init_routes(app):
    """Initialize all routes for the Flask app."""
    
    @app.route('/upload', methods=['POST'])
    def upload_images():
        """
        Handle image uploads to create a face illusion.
        
        Expects two images in the request:
            - 'front': Front-facing portrait image
            - 'side': Side profile image
        
        Returns:
            JSON response with:
                - mergedImage: Base64-encoded face illusion image
        """
        if 'front' not in request.files or 'side' not in request.files:
            return jsonify({'error': 'Missing files, both front and side images are required'}), 400

        processor = ImageProcessor()

        try:
            # Get files
            front_file = request.files['front']
            side_file = request.files['side']
            
            # Create folders for output if they don't exist
            os.makedirs(os.path.dirname(FRONT_NO_BG_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(SIDE_NO_BG_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(MERGED_IMAGE_PATH), exist_ok=True)
            
            # Process images and create illusion
            logger.info("Processing images to create face illusion")
            
            # Make copies of the file objects since we'll need to reuse them
            front_file_copy = front_file
            side_file_copy = side_file
            
            # Use the simplified pipeline from the improved ImageProcessor
            merged_base64 = processor.process_face_illusion(front_file, side_file)
            
            # For debugging/development: save intermediate images
            # We need to reset file pointers or create new file objects
            front_file_copy.seek(0)
            side_file_copy.seek(0)
            
            front_img = processor.decode_image(front_file_copy)
            side_img = processor.decode_image(side_file_copy)
            
            # Remove backgrounds and save
            front_no_bg = processor.remove_background(front_img)
            side_no_bg = processor.remove_background(side_img)
            cv2.imwrite(FRONT_NO_BG_PATH, front_no_bg)
            cv2.imwrite(SIDE_NO_BG_PATH, side_no_bg)
            
            # Save final merged image
            merged_image_data = base64.b64decode(merged_base64)
            with open(MERGED_IMAGE_PATH, 'wb') as f:
                f.write(merged_image_data)
                
            # Prepare response
            response = {
                'mergedImage': merged_base64,
                'status': 'success',
                'message': 'Face illusion created successfully'
            }
            
            logger.info("Successfully created face illusion")
            return jsonify(response)

        except Exception as e:
            logger.error(f"Error in /upload endpoint: {str(e)}")
            return jsonify({
                'error': f'Failed to process images: {str(e)}',
                'status': 'error'
            }), 500

    @app.route('/adjust', methods=['POST'])
    def adjust_images():
        """
        Adjust the face illusion by setting custom parameters.
        
        Expected JSON body:
            - grayscale: Convert to grayscale (optional)
            - flip_side: Flip the side profile image (optional)
        
        Returns:
            JSON response with:
                - mergedImage: Base64-encoded adjusted face illusion
        """
        try:
            data = request.json or {}
            processor = ImageProcessor()
            
            # Default values
            grayscale = data.get('grayscale', False)
            flip_side = data.get('flip_side', False)
            
            # Read images
            front_img = cv2.imread(FRONT_NO_BG_PATH, cv2.IMREAD_UNCHANGED)
            side_img = cv2.imread(SIDE_NO_BG_PATH, cv2.IMREAD_UNCHANGED)
            
            if front_img is None or side_img is None:
                return jsonify({
                    'error': 'Failed to load images. Make sure to upload images first.',
                    'status': 'error'
                }), 400
                
            # Ensure alpha channel
            front_img = processor.ensure_alpha_channel(front_img)
            side_img = processor.ensure_alpha_channel(side_img)
            
            # Flip side image if requested
            if flip_side:
                side_img = cv2.flip(side_img, 1)
            
            # Create face illusion
            merged_image = processor.create_face_illusion(front_img, side_img)
            
            # Convert to grayscale if requested
            if grayscale:
                # Extract RGB channels and convert to grayscale
                bgr = merged_image[:, :, :3]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                # Create grayscale with alpha
                merged_gray = np.zeros((merged_image.shape[0], merged_image.shape[1], 4), dtype=np.uint8)
                merged_gray[:, :, 0] = gray
                merged_gray[:, :, 1] = gray
                merged_gray[:, :, 2] = gray
                merged_gray[:, :, 3] = merged_image[:, :, 3]  # Preserve alpha channel
                merged_image = merged_gray
            
            # Save adjusted image
            cv2.imwrite(MERGED_IMAGE_PATH, merged_image)
            
            # Return base64
            merged_base64 = processor.encode_image_to_base64(merged_image)
            
            return jsonify({
                'mergedImage': merged_base64,
                'status': 'success',
                'message': 'Face illusion adjusted successfully'
            })
            
        except Exception as e:
            logger.error(f"Error in /adjust endpoint: {str(e)}")
            return jsonify({
                'error': f'Failed to adjust images: {str(e)}',
                'status': 'error'
            }), 500
            
    @app.route('/health', methods=['GET'])
    def health_check():
        """Simple health check endpoint to verify API is running."""
        return jsonify({
            'status': 'healthy',
            'message': 'Face illusion API is running'
        })