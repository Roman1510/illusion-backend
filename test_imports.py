import flask
import cv2
import mediapipe
import rembg
import numpy
import PIL

print(f"Flask version: {flask.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Mediapipe version: {mediapipe.__version__}")
print(f"rembg version: {rembg.__version__}")  # Note: rembg might not have __version__
print(f"NumPy version: {numpy.__version__}")
print(f"Pillow version: {PIL.__version__}")
print("All libraries imported successfully!")