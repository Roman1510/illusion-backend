import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
import logging
import base64

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Streamlined processor for creating face illusions by merging front and side profile images."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        logger.info("ImageProcessor initialized")

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
            result = remove(image)
            if result.shape[2] == 3:
                result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            return result
        except Exception as e:
            logger.error(f"Error removing background: {str(e)}")
            raise

    def detect_face_landmarks(self, image: np.ndarray):
        """Detect facial landmarks using MediaPipe and determine face orientation."""
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            logger.warning("No face landmarks detected")
            return None, "unknown"
        
        landmarks = results.multi_face_landmarks[0]
        
        # Key points for orientation detection
        nose_tip = (int(landmarks.landmark[1].x * width), int(landmarks.landmark[1].y * height))
        left_eye = (int(landmarks.landmark[33].x * width), int(landmarks.landmark[33].y * height))
        right_eye = (int(landmarks.landmark[263].x * width), int(landmarks.landmark[263].y * height))
        left_cheek = (int(landmarks.landmark[234].x * width), int(landmarks.landmark[234].y * height))
        right_cheek = (int(landmarks.landmark[454].x * width), int(landmarks.landmark[454].y * height))
        
        # Calculate asymmetry to determine orientation
        left_dist = abs(nose_tip[0] - left_cheek[0])
        right_dist = abs(nose_tip[0] - right_cheek[0])
        
        # Determine orientation
        orientation = "front"
        if left_dist / max(1, right_dist) < 0.8:
            orientation = "right"  # Face looking to the right
        elif right_dist / max(1, left_dist) < 0.8:
            orientation = "left"   # Face looking to the left
        
        return landmarks, orientation

    def align_face_features(self, front_image: np.ndarray, side_image: np.ndarray):
        """Align front and side faces based on facial features for better blending."""
        # Get landmarks and orientations
        front_landmarks, front_orientation = self.detect_face_landmarks(front_image)
        side_landmarks, side_orientation = self.detect_face_landmarks(side_image)
        
        # If can't detect landmarks in either image, use basic alignment
        if front_landmarks is None or side_landmarks is None:
            logger.warning("Using basic alignment due to missing landmarks")
            # Resize to consistent size
            height = max(front_image.shape[0], side_image.shape[0])
            width = max(front_image.shape[1], side_image.shape[1])
            front_aligned = cv2.resize(front_image, (width, height))
            side_aligned = cv2.resize(side_image, (width, height))
            return front_aligned, side_aligned, front_orientation, side_orientation
        
        # Calculate key dimensions from front image
        front_h, front_w = front_image.shape[:2]
        side_h, side_w = side_image.shape[:2]
        
        # Extract facial feature points from landmarks
        front_eye_y = int(front_landmarks.landmark[33].y * front_h)
        front_mouth_y = int(front_landmarks.landmark[13].y * front_h)
        front_face_height = front_mouth_y - front_eye_y
        
        side_eye_y = int(side_landmarks.landmark[33].y * side_h)
        side_mouth_y = int(side_landmarks.landmark[13].y * side_h)
        side_face_height = side_mouth_y - side_eye_y
        
        # Calculate scale ratio to match face heights
        scale_ratio = 1.0
        if side_face_height > 0 and front_face_height > 0:
            scale_ratio = front_face_height / side_face_height
        
        # Resize side image to match facial feature scale
        new_side_h = int(side_h * scale_ratio)
        new_side_w = int(side_w * scale_ratio)
        side_aligned = cv2.resize(side_image, (new_side_w, new_side_h))
        
        # Resize front image to higher resolution if needed
        target_resolution = 800
        if front_w < target_resolution or front_h < target_resolution:
            scale = max(target_resolution / front_w, target_resolution / front_h)
            new_front_w = int(front_w * scale)
            new_front_h = int(front_h * scale)
            front_aligned = cv2.resize(front_image, (new_front_w, new_front_h))
        else:
            front_aligned = front_image.copy()
        
        # Ensure both images are the same size after alignment
        max_h = max(front_aligned.shape[0], side_aligned.shape[0])
        max_w = max(front_aligned.shape[1], side_aligned.shape[1])
        
        # Create new canvases to hold the aligned images
        front_canvas = np.zeros((max_h, max_w, 4), dtype=np.uint8)
        side_canvas = np.zeros((max_h, max_w, 4), dtype=np.uint8)
        
        # Center the images on the canvas
        front_y_offset = (max_h - front_aligned.shape[0]) // 2
        front_x_offset = (max_w - front_aligned.shape[1]) // 2
        front_canvas[
            front_y_offset:front_y_offset + front_aligned.shape[0],
            front_x_offset:front_x_offset + front_aligned.shape[1]
        ] = front_aligned
        
        side_y_offset = (max_h - side_aligned.shape[0]) // 2
        side_x_offset = (max_w - side_aligned.shape[1]) // 2
        side_canvas[
            side_y_offset:side_y_offset + side_aligned.shape[0],
            side_x_offset:side_x_offset + side_aligned.shape[1]
        ] = side_aligned
        
        return front_canvas, side_canvas, front_orientation, side_orientation

    def create_face_illusion(self, front_image: np.ndarray, side_image: np.ndarray) -> np.ndarray:
        """Create an optical illusion by merging front and side face images."""
        try:
            # Align images and get orientations
            front_aligned, side_aligned, front_orientation, side_orientation = self.align_face_features(front_image, side_image)
            
            # Determine which side profile to use based on orientations
            if side_orientation == "left":
                # We want the side profile to face right for better blending
                side_aligned = cv2.flip(side_aligned, 1)
                logger.info("Flipped side image to face right")
                side_side = "right"  # After flipping, it's now facing right
            else:
                # Already facing right or unknown (assume right)
                side_side = "right"
            
            # Create the merged illusion
            return self.merge_images_seamlessly(front_aligned, side_aligned, side_side)
            
        except Exception as e:
            logger.error(f"Error creating face illusion: {str(e)}")
            raise

    def merge_images_seamlessly(self, front_image: np.ndarray, side_image: np.ndarray, side_side: str = "right") -> np.ndarray:
        """Merge front and side images with better seam blending for a natural illusion."""
        height, width = front_image.shape[:2]
        midline = width // 2
        
        # Create output image with alpha channel
        merged_image = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Decide which half of front image to use
        # We want the side opposite to the profile side
        if side_side == "right":  # Side profile is facing right
            # Use left half of front image
            front_half = front_image[:, :midline]
            side_half = side_image[:, midline:]
            
            # Copy halves
            merged_image[:, :midline] = front_half
            merged_image[:, midline:] = side_half
            
            # Define blend region
            blend_width = width // 20  # 5% of width for subtle blending
            blend_start = max(0, midline - blend_width)
            blend_end = min(width, midline + blend_width)
            
        else:  # Side profile is facing left
            # Use right half of front image
            front_half = front_image[:, midline:]
            side_half = side_image[:, :midline]
            
            # Copy halves
            merged_image[:, midline:] = front_half
            merged_image[:, :midline] = side_half
            
            # Define blend region
            blend_width = width // 20  # 5% of width for subtle blending
            blend_start = max(0, midline - blend_width)
            blend_end = min(width, midline + blend_width)
        
        # Apply gradient blending for smoother transition
        for x in range(blend_start, blend_end):
            alpha = (x - blend_start) / (blend_end - blend_start)
            merged_image[:, x] = cv2.addWeighted(front_image[:, x], 1-alpha, side_image[:, x], alpha, 0)
        
        # Ensure alpha channel is preserved
        alpha_mask = np.logical_or(front_image[:, :, 3] > 0, side_image[:, :, 3] > 0).astype(np.uint8) * 255
        merged_image[:, :, 3] = alpha_mask
        
        return merged_image

    def process_face_illusion(self, front_file, side_file) -> str:
        """Process front and side images to create a face illusion and return base64 encoded result."""
        try:
            # Decode images
            front_image = self.decode_image(front_file)
            side_image = self.decode_image(side_file)
            
            # Remove backgrounds
            front_no_bg = self.remove_background(front_image)
            side_no_bg = self.remove_background(side_image)
            
            # Create illusion
            result_image = self.create_face_illusion(front_no_bg, side_no_bg)
            
            # Encode to base64
            return self.encode_image_to_base64(result_image)
            
        except Exception as e:
            logger.error(f"Error in face illusion processing pipeline: {str(e)}")
            raise

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode an image to a base64 string with transparency support."""
        try:
            _, encoded_image = cv2.imencode('.png', image)
            return base64.b64encode(encoded_image).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            raise