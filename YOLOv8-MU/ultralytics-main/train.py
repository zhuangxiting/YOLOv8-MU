from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('/root/autodl-tmp/JX/YOLOV8/1/swin+USE+MPDIOU+CretoNext/ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml').load('/root/autodl-tmp/JX/YOLOV8/1/swin+USE+MPDIOU+CretoNext/ultralytics-main/yolov8n.pt')

# Load a pretrained YOLO model (recommended for training)
# model = YOLO('/home/ubuntu/Desktop/JX/YOLOv8/ultralytics-main/ultralytics-main/yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='/root/autodl-tmp/JX/YOLOV8/1/swin+USE+MPDIOU+CretoNext/ultralytics-main/ultralytics/cfg/datasets/urpc.yaml', epochs=100)

# # Evaluate the model's performance on the validation set
# results = model.val()
#
# # Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')
#
# # Export the model to ONNX format
# success = model.export(format='onnx')
