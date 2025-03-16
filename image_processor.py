import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from typing import Tuple, Dict
import logging
import base64

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Class to handle image processing tasks like decoding, background removal, landmark detection, and merging."""
    
    def __init__(self):
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
            result = remove(image)
            if result.shape[2] == 3:
                result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            return result
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
        """Detect facial landmarks using MediaPipe with fallback for side profiles."""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            landmarks_dict = {"left_eye": None, "right_eye": None, "mouth_center": None, "nose_tip": None, "orientation": "unknown"}
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                landmarks_dict["left_eye"] = (int(landmarks[33].x * image_width), int(landmarks[33].y * image_height))
                landmarks_dict["right_eye"] = (int(landmarks[263].x * image_width), int(landmarks[263].y * image_height))
                landmarks_dict["nose_tip"] = (int(landmarks[1].x * image_width), int(landmarks[1].y * image_height))
                landmarks_dict["mouth_center"] = (int(landmarks[13].x * image_width), int(landmarks[13].y * image_height))
                logger.info("Landmarks detected successfully")
                if is_side_profile and landmarks_dict["nose_tip"]:
                    nose_x = landmarks_dict["nose_tip"][0]
                    if nose_x < image_width // 2:
                        landmarks_dict["orientation"] = "left"
                    else:
                        landmarks_dict["orientation"] = "right"
            else:
                logger.warning("No face landmarks detected in image")
                if is_side_profile:
                    nose_tip = self.estimate_nose_tip(image)
                    landmarks_dict["nose_tip"] = nose_tip
                    nose_x = nose_tip[0]
                    if nose_x < image_width // 2:
                        landmarks_dict["orientation"] = "left"
                    else:
                        landmarks_dict["orientation"] = "right"
                else:
                    nose_tip = self.estimate_nose_tip(image)
                    landmarks_dict["nose_tip"] = nose_tip
            return landmarks_dict
        except Exception as e:
            logger.error(f"Error detecting landmarks: {str(e)}")
            raise

    def crop_to_face(self, image: np.ndarray, landmarks: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """Crop the image to the face and neck area dynamically based on landmarks and image dimensions."""
        try:
            height, width = image.shape[:2]
            nose_tip = landmarks.get("nose_tip")
            if not nose_tip:
                logger.warning("Nose tip not detected, returning original image")
                return image

            nose_y = nose_tip[1]
            eye_y = nose_y - int(height * 0.15) if landmarks.get("left_eye") is None else landmarks["left_eye"][1]
            mouth_y = nose_y + int(height * 0.2) if landmarks.get("mouth_center") is None else landmarks["mouth_center"][1]

            face_height = mouth_y - eye_y
            top = max(0, eye_y - int(face_height * 0.5))
            bottom = min(height, mouth_y + int(face_height * 1.0))

            if landmarks.get("left_eye") and landmarks.get("right_eye"):
                eye_distance = abs(landmarks["right_eye"][0] - landmarks["left_eye"][0])
                face_width = eye_distance * 2
            else:
                face_width = int(width * 0.5)

            left = max(0, nose_tip[0] - face_width // 2)
            right = min(width, nose_tip[0] + face_width // 2)

            cropped_image = image[top:bottom, left:right]
            return cropped_image
        except Exception as e:
            logger.error(f"Error cropping image: {str(e)}")
            raise

    def align_images(self, front_image: np.ndarray, side_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale images to match a common resolution while preserving aspect ratio."""
        try:
            front_height, front_width = front_image.shape[:2]
            side_height, side_width = side_image.shape[:2]

            # Calculate aspect ratios
            front_aspect = front_width / front_height
            side_aspect = side_width / side_height

            # Use the front image as the reference for target dimensions
            target_height = front_height
            target_width = front_width

            # Scale front image if needed (typically not necessary since it's the reference)
            if (front_width, front_height) != (target_width, target_height):
                front_image = cv2.resize(front_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # Scale side image while preserving aspect ratio
            if side_aspect > front_aspect:
                # Side image is wider, scale to match height
                new_side_height = target_height
                new_side_width = int(new_side_height * side_aspect)
            else:
                # Side image is taller, scale to match width
                new_side_width = target_width
                new_side_height = int(new_side_width / side_aspect)

            side_image = cv2.resize(side_image, (new_side_width, new_side_height), interpolation=cv2.INTER_AREA)

            # Handle dimensions: pad if smaller, crop if larger
            if new_side_width < target_width or new_side_height < target_height:
                # Pad the side image to match target dimensions
                top_pad = max(0, (target_height - new_side_height) // 2)
                bottom_pad = max(0, target_height - new_side_height - top_pad)
                left_pad = max(0, (target_width - new_side_width) // 2)
                right_pad = max(0, target_width - new_side_width - left_pad)
                side_image = cv2.copyMakeBorder(
                    side_image, top_pad, bottom_pad, left_pad, right_pad,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
                )
            else:
                # Crop the side image to match target dimensions
                x_offset = max(0, (new_side_width - target_width) // 2)
                y_offset = max(0, (new_side_height - target_height) // 2)
                side_image = side_image[
                    y_offset:y_offset + target_height,
                    x_offset:x_offset + target_width
                ]

            return front_image, side_image
        except Exception as e:
            logger.error(f"Error aligning images: {str(e)}")
            raise

    def merge_images(self, front_image: np.ndarray, side_image: np.ndarray, front_landmarks: Dict, side_landmarks: Dict) -> np.ndarray:
        """Merge the front and side images with a smooth blend based on orientation."""
        try:
            front_image = self.crop_to_face(front_image, front_landmarks)
            side_image = self.crop_to_face(side_image, side_landmarks)
            front_image, side_image = self.align_images(front_image, side_image)
            if front_image.shape[2] == 3:
                front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2BGRA)
            if side_image.shape[2] == 3:
                side_image = cv2.cvtColor(side_image, cv2.COLOR_BGR2BGRA)

            height, width = front_image.shape[:2]
            midline = width // 2
            merged_image = np.zeros((height, width, 4), dtype=np.uint8)

            orientation = side_landmarks.get("orientation", "left")  # Default to left if unknown
            if orientation == "left":
                # Side profile is facing left, place on the left side
                side_image_mirrored = side_image  # No flip needed since it's facing left
                # Left half from side image
                merged_image[:, :midline] = side_image_mirrored[:, :midline]
                # Right half from front image
                merged_image[:, midline:] = front_image[:, midline:]
                # Blend the transition
                blend_width = int(width * 0.05)  # Increased for smoother blending
                for x in range(max(0, midline - blend_width), min(width, midline + blend_width)):
                    alpha = (x - (midline - blend_width)) / (2 * blend_width)
                    alpha = max(0, min(1, alpha))
                    merged_image[:, x] = (
                        (1 - alpha) * side_image_mirrored[:, x] + alpha * front_image[:, x]
                    ).astype(np.uint8)
            elif orientation == "right":
                # Side profile is facing right, place on the right side
                side_image_mirrored = side_image  # No flip needed since it's facing right
                # Left half from front image
                merged_image[:, :midline] = front_image[:, :midline]
                # Right half from side image
                merged_image[:, midline:] = side_image_mirrored[:, midline:]
                # Blend the transition
                blend_width = int(width * 0.05)  # Increased for smoother blending
                for x in range(max(0, midline - blend_width), min(width, midline + blend_width)):
                    alpha = (x - (midline - blend_width)) / (2 * blend_width)
                    alpha = max(0, min(1, alpha))
                    merged_image[:, x] = (
                        (1 - alpha) * front_image[:, x] + alpha * side_image_mirrored[:, x]
                    ).astype(np.uint8)

            # Ensure transparency where no content exists
            mask = (merged_image[:, :, 3] == 0)
            merged_image[mask] = [0, 0, 0, 0]

            return merged_image
        except Exception as e:
            logger.error(f"Error merging images: {str(e)}")
            raise

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode an image to a base64 string with transparency support."""
        try:
            _, encoded_image = cv2.imencode('.png', image)
            return base64.b64encode(encoded_image).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            raise