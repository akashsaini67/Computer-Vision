from ultralytics import YOLO
import requests
import cv2
import matplotlib.pyplot as plt

# Image URL
image_url = "https://ultralytics.com/images/bus.jpg"

# Download image
img_data = requests.get(image_url).content
with open("test.jpg", "wb") as f:
    f.write(img_data)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Perform detection
results = model("test.jpg")

# Show result
import cv2
import matplotlib.pyplot as plt

annotated_image = results[0].plot()
annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,8))
plt.imshow(annotated_image)
plt.axis("off")
plt.show()

# Print detected objects
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        print(f"Object: {model.names[cls]} | Confidence: {conf:.2f}")