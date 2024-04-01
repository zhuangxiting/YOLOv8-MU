from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("model.pt")#使用最好的模型
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0")#摄像头检测
results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments
