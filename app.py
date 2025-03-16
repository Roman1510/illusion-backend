import os
import base64
import logging
from typing import Tuple, Dict, List, Optional
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Directory to store temporary images
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# File paths for temporary storage
FRONT_NO_BG_PATH = os.path.join(TEMP_DIR, "front_no_bg.png")
SIDE_NO_BG_PATH = os.path.join(TEMP_DIR, "side_no_bg.png")
MERGED_IMAGE_PATH = os.path.join(TEMP_DIR, "merged_image.png")

class ImageProcessor:
    """Class to handle image processing tasks like decoding, background removal, landmark detection, and merging."""
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh for landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        logger.info("ImageProcessor initialized with MediaPipe Face Mesh")

    def decode_image(self, file) -> np.ndarray:
        """Decode an uploaded image file into a NumPy array."""
        try:
            image_data = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            raise

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """Remove the background from an image using rembg."""
        try:
            return remove(image)
        except Exception as e:
            logger.error(f"Error removing background: {str(e)}")
            raise

    def estimate_nose_tip(self, image: np.ndarray) -> Tuple[int, int]:
        """Estimate the nose tip position for a side profile image if landmarks aren't detected."""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.warning("No contours found, using default nose tip position")
                return (width // 2, height // 2)

            face_contour = max(contours, key=cv2.contourArea)
            rightmost = max(face_contour, key=lambda point: point[0][0])
            nose_x, nose_y = rightmost[0]
            nose_y = height // 2
            return (nose_x, nose_y)
        except Exception as e:
            logger.error(f"Error estimating nose tip: {str(e)}")
            return (image.shape[1] // 2, image.shape[0] // 2)

    def detect_landmarks(self, image: np.ndarray, image_width: int, image_height: int, is_side_profile: bool = False) -> Dict[str, Tuple[int, int]]:
        """Detect facial landmarks (eyes, mouth, nose) using MediaPipe, with fallback for side profiles and orientation detection."""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            landmarks_dict = {
                "left_eye": None,
                "right_eye": None,
                "mouth_center": None,
                "nose_tip": None,
                "orientation": "unknown"  # Add orientation field
            }

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                landmarks_dict["left_eye"] = (
                    int(landmarks[33].x * image_width),
                    int(landmarks[33].y * image_height)
                )
                landmarks_dict["right_eye"] = (
                    int(landmarks[263].x * image_width),
                    int(landmarks[263].y * image_height)
                )
                landmarks_dict["nose_tip"] = (
                    int(landmarks[1].x * image_width),
                    int(landmarks[1].y * image_height)
                )
                landmarks_dict["mouth_center"] = (
                    int(landmarks[13].x * image_width),
                    int(landmarks[13].y * image_height)
                )
                logger.info("Landmarks detected successfully")
                if is_side_profile and landmarks_dict["nose_tip"] and landmarks_dict["right_eye"]:
                    # Determine orientation: if nose tip is right of right eye, it's a left profile
                    if landmarks_dict["nose_tip"][0] > landmarks_dict["right_eye"][0]:
                        landmarks_dict["orientation"] = "left"
                    else:
                        landmarks_dict["orientation"] = "right"
            else:
                logger.warning("No face landmarks detected in image")
                if is_side_profile:
                    nose_tip = self.estimate_nose_tip(image)
                    landmarks_dict["nose_tip"] = nose_tip
                    logger.info(f"Estimated nose tip for side profile at {nose_tip}")
                    # For estimated landmarks, assume orientation based on nose tip position
                    if nose_tip[0] > image_width // 2:
                        landmarks_dict["orientation"] = "left"
                    else:
                        landmarks_dict["orientation"] = "right"
                else:
                    nose_tip = self.estimate_nose_tip(image)
                    landmarks_dict["nose_tip"] = nose_tip
                    logger.info(f"Estimated nose tip for front image at {nose_tip}")

            return landmarks_dict
        except Exception as e:
            logger.error(f"Error detecting landmarks: {str(e)}")
            raise

    def crop_to_face(self, image: np.ndarray, landmarks: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """Crop the image to the face and neck area with improved symmetry."""
        try:
            height, width = image.shape[:2]
            nose_tip = landmarks.get("nose_tip")

            if not nose_tip:
                logger.warning("Nose tip not detected, returning original image")
                return image

            nose_y = nose_tip[1]
            eye_y = nose_y - 50 if landmarks.get("left_eye") is None else landmarks["left_eye"][1]
            mouth_y = nose_y + 50 if landmarks.get("mouth_center") is None else landmarks["mouth_center"][1]

            # Use a wider crop to ensure the front image half is not too narrow
            top = max(0, eye_y - 70)
            bottom = min(height, mouth_y + 120)
            left = max(0, nose_tip[0] - 150)
            right = min(width, nose_tip[0] + 150)

            cropped_image = image[top:bottom, left:right]
            return cropped_image
        except Exception as e:
            logger.error(f"Error cropping image: {str(e)}")
            raise

    def align_images(self, front_image: np.ndarray, side_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rescale the side image to match the front image's dimensions without altering scale."""
        try:
            front_height, front_width = front_image.shape[:2]
            side_image = cv2.resize(side_image, (front_width, front_height))  # No scale adjustment
            return front_image, side_image
        except Exception as e:
            logger.error(f"Error aligning images: {str(e)}")
            raise

    def merge_images(self, front_image: np.ndarray, side_image: np.ndarray, front_landmarks: Dict, side_landmarks: Dict) -> np.ndarray:
        """Merge the front and side images, placing the side contour on the left and front image on the right with a clean split."""
        try:
            # Crop images to face and neck area
            front_image = self.crop_to_face(front_image, front_landmarks)
            side_image = self.crop_to_face(side_image, side_landmarks)

            # Rescale and align images
            front_image, side_image = self.align_images(front_image, side_image)

            # Ensure images are in RGBA format for alpha blending and transparency
            if front_image.shape[2] == 3:
                front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2BGRA)
            if side_image.shape[2] == 3:
                side_image = cv2.cvtColor(side_image, cv2.COLOR_BGR2BGRA)

            # Get dimensions of the front image (which is the base)
            height, width = front_image.shape[:2]

            # Recalculate midline based on image center
            midline = width // 2

            # Create the merged image with transparent background
            merged_image = np.zeros((height, width, 4), dtype=np.uint8)
            merged_image[:, :, 3] = 0  # Initialize with transparent alpha channel

            # Determine side profile orientation and adjust merging
            orientation = side_landmarks.get("orientation", "unknown")
            if orientation == "left":
                # Left profile (right-facing): mirror the side image and place on the left half
                side_image_mirrored = cv2.flip(side_image, 1)
                merged_image[:, :midline] = side_image_mirrored[:, :midline]
                merged_image[:, midline:] = front_image[:, midline:]
            elif orientation == "right":
                # Right profile (left-facing): place on the left half (no mirroring needed)
                merged_image[:, :midline] = side_image[:, :midline]
                merged_image[:, midline:] = front_image[:, midline:]
            else:
                logger.warning("Could not determine side profile orientation, defaulting to left profile")
                side_image_mirrored = cv2.flip(side_image, 1)
                merged_image[:, :midline] = side_image_mirrored[:, :midline]
                merged_image[:, midline:] = front_image[:, midline:]

            # Apply a narrow alpha blending at the midline for a smooth transition
            blend_width = int(width * 0.02)  # 2% for a tight blend
            for x in range(max(0, midline - blend_width), min(width, midline + blend_width)):
                alpha = (x - (midline - blend_width)) / (2 * blend_width)
                alpha = max(0, min(1, alpha))
                if orientation == "left":
                    merged_image[:, x] = (1 - alpha) * side_image_mirrored[:, x] + alpha * front_image[:, x]
                elif orientation == "right":
                    merged_image[:, x] = (1 - alpha) * side_image[:, x] + alpha * front_image[:, x]

            # Ensure transparency where no content exists
            mask = (merged_image[:, :, 3] == 0)
            merged_image[mask] = [0, 0, 0, 0]  # Keep transparent

            return merged_image
        except Exception as e:
            logger.error(f"Error merging images: {str(e)}")
            raise

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode an image to a base64 string with transparency support."""
        try:
            _, encoded_image = cv2.imencode('.png', image)  # PNG supports transparency
            return base64.b64encode(encoded_image).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            raise

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

        # Resize images for consistency (this resize happens before cropping)
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

        # Convert to grayscale to match the example
        if merged_image.shape[2] == 4:
            merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGRA2GRAY)
            merged_image = cv2.cvtColor(merged_image, cv2.COLOR_GRAY2BGRA)  # Preserve alpha channel

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)