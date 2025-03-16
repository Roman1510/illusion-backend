import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
import logging
import base64

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Minimal and efficient processor for creating face illusions."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def decode_image(self, file) -> np.ndarray:
        """Decode an uploaded image file into a NumPy array."""
        try:
            image_data = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) if image is not None else None
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            raise
    
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """Remove the background from an image."""
        result = remove(image)
        return cv2.cvtColor(result, cv2.COLOR_BGR2BGRA) if result.shape[2] == 3 else result
    
    def process_face_illusion(self, front_file, side_file) -> str:
        """Full pipeline: process images to create illusion and return base64 result."""
        # Decode and remove backgrounds
        front_image = self.remove_background(self.decode_image(front_file))
        side_file.seek(0)  # Reset file pointer
        side_image = self.remove_background(self.decode_image(side_file))
        
        # Process and create illusion
        result = self.create_illusion(front_image, side_image)
        
        # Encode to base64
        _, encoded = cv2.imencode('.png', result)
        return base64.b64encode(encoded).decode('utf-8')
    
    def create_illusion(self, front_image: np.ndarray, side_image: np.ndarray) -> np.ndarray:
        """Create the face illusion by analyzing and merging images."""
        # Get face orientations and align images
        front_aligned, side_aligned, side_direction = self.align_images(front_image, side_image)
        
        # Create final merged image
        return self.blend_images(front_aligned, side_aligned, side_direction)
    
    def align_images(self, front_image: np.ndarray, side_image: np.ndarray):
        """Analyze faces, determine orientations, and align images."""
        # Detect landmarks and determine orientation
        front_orientation, front_landmarks = self.analyze_face(front_image)
        side_orientation, side_landmarks = self.analyze_face(side_image)
        
        # Flip side image if needed for consistent orientation
        if side_orientation == "left":
            side_image = cv2.flip(side_image, 1)
            side_orientation = "right"
        
        # Match face proportions if landmarks detected
        if front_landmarks and side_landmarks:
            # Extract key facial dimensions
            front_h, front_w = front_image.shape[:2]
            side_h, side_w = side_image.shape[:2]
            
            # Calculate face height ratio using eye-to-mouth distance
            front_eye_y = int(front_landmarks.landmark[33].y * front_h)
            front_mouth_y = int(front_landmarks.landmark[13].y * front_h)
            front_face_h = front_mouth_y - front_eye_y
            
            side_eye_y = int(side_landmarks.landmark[33].y * side_h)
            side_mouth_y = int(side_landmarks.landmark[13].y * side_h)
            side_face_h = side_mouth_y - side_eye_y
            
            if side_face_h > 0 and front_face_h > 0:
                # Scale side image to match front face proportions
                scale = front_face_h / side_face_h
                new_h = int(side_h * scale)
                new_w = int(side_w * scale)
                side_image = cv2.resize(side_image, (new_w, new_h))
        
        # Ensure both images are the same size
        max_h = max(front_image.shape[0], side_image.shape[0])
        max_w = max(front_image.shape[1], side_image.shape[1])
        
        front_resized = cv2.resize(front_image, (max_w, max_h))
        side_resized = cv2.resize(side_image, (max_w, max_h))
        
        return front_resized, side_resized, side_orientation
    
    def analyze_face(self, image: np.ndarray):
        """Detect facial landmarks and determine face orientation."""
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return "unknown", None
        
        landmarks = results.multi_face_landmarks[0]
        
        # Determine orientation based on nose and cheek positions
        nose_x = landmarks.landmark[1].x * w
        left_cheek_x = landmarks.landmark[234].x * w
        right_cheek_x = landmarks.landmark[454].x * w
        
        left_dist = abs(nose_x - left_cheek_x)
        right_dist = abs(nose_x - right_cheek_x)
        
        # Threshold for determining orientation
        if left_dist / max(1, right_dist) < 0.8:
            orientation = "right"
        elif right_dist / max(1, left_dist) < 0.8:
            orientation = "left"
        else:
            orientation = "front"
            
        return orientation, landmarks
    
    def blend_images(self, front_image: np.ndarray, side_image: np.ndarray, side_direction: str) -> np.ndarray:
        """Blend front and side images with smooth transition."""
        h, w = front_image.shape[:2]
        midline = w // 2
        result = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Copy the halves
        result[:, :midline] = front_image[:, :midline]
        result[:, midline:] = side_image[:, midline:]
        
        # Create a smooth transition across the midline
        blend_width = w // 20  # 5% of width
        for x in range(max(0, midline - blend_width), min(w, midline + blend_width)):
            # Calculate blend factor with cosine interpolation
            alpha = 0.5 - 0.5 * np.cos(np.pi * (x - (midline - blend_width)) / (2 * blend_width))
            alpha = max(0, min(1, alpha))
            result[:, x] = ((1 - alpha) * front_image[:, x] + alpha * side_image[:, x]).astype(np.uint8)
        
        # Ensure proper alpha channel
        alpha_mask = np.logical_or(front_image[:, :, 3] > 0, side_image[:, :, 3] > 0).astype(np.uint8) * 255
        result[:, :, 3] = alpha_mask
        
        return result