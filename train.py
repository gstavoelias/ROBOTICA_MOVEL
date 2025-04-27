from ultralytics import YOLO

# Carregar o modelo
model = YOLO('yolov8s.pt')

# # Treinar o modelo
# model.train(
#     data='datasets/content/data.yaml',
#     epochs=60,
#     imgsz=640
# )
# model.predict(source="IMG2.jpg", conf=0.25, save=True, classes=[0])