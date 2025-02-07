import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np

# Initialize the object detection model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Input image path
input_image_path = r"C:\Users\meghna\Downloads\path pic.jpg"  # Replace with your image path
output_image_path = r"C:\Users\meghna\Downloads\output.png"  # Path to save the result

# Load the image
image = cv2.imread(input_image_path)
if image is None:
    print("Error: Could not read the image.")
    exit()

# Convert the image to a tensor
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

# Perform object detection
with torch.no_grad():
    predictions = model(image_tensor)

# Extract predictions
boxes = predictions[0]['boxes'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()
threshold = 0.5  # Confidence threshold
detected_objects = boxes[scores > threshold]

# Define the simulated agent
height, width, _ = image.shape
agent_x = width // 2
agent_y = height - 50
agent_width = 50
agent_height = 50

# Draw detected objects and simulate avoidance logic
avoid_left = avoid_right = False
for box in detected_objects:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

    # Check if the object is in the agent's path
    if agent_y < y2 < agent_y + agent_height and (
        (x1 < agent_x < x2) or (x1 < agent_x + agent_width < x2)
    ):
        if x1 < agent_x:
            avoid_right = True  # Obstacle to the left
        elif x2 > agent_x + agent_width:
            avoid_left = True  # Obstacle to the right

# Determine movement direction
direction = "forward"
if avoid_left and not avoid_right:
    direction = "right"
elif avoid_right and not avoid_left:
    direction = "left"
elif avoid_left and avoid_right:
    direction = "backward"

# Draw the agent
cv2.rectangle(image, (agent_x, agent_y), (agent_x + agent_width, agent_y + agent_height), (0, 0, 255), -1)

# Draw direction arrow
arrow_color = (255, 0, 0)  # Blue arrow
if direction == "forward":
    cv2.arrowedLine(image, (agent_x + agent_width // 2, agent_y), (agent_x + agent_width // 2, agent_y - 50), arrow_color, 3)
elif direction == "right":
    cv2.arrowedLine(image, (agent_x + agent_width // 2, agent_y + agent_height // 2), (agent_x + agent_width // 2 + 50, agent_y + agent_height // 2), arrow_color, 3)
elif direction == "left":
    cv2.arrowedLine(image, (agent_x + agent_width // 2, agent_y + agent_height // 2), (agent_x + agent_width // 2 - 50, agent_y + agent_height // 2), arrow_color, 3)
elif direction == "backward":
    cv2.arrowedLine(image, (agent_x + agent_width // 2, agent_y + agent_height), (agent_x + agent_width // 2, agent_y + agent_height + 50), arrow_color, 3)

# Save and display the result
cv2.imwrite(output_image_path, image)
cv2.imshow("Object Detection with Avoidance Logic", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Processed image saved to: {output_image_path}")
