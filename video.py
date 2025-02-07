import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Initialize the object detection model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Path to the input video
input_video_path =r"C:\Users\meghna\Downloads\video.mp4"  # Replace with your video file path
output_video_path =r"C:\Users\meghna\Downloads\output.m4a"   # Output video path

# Open the video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up the video writer for saving the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Perform object detection
    with torch.no_grad():
        predictions = model(frame_tensor)

    # Annotate the frame with bounding boxes
    for box, label, score in zip(
        predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']
    ):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.tolist())
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label and confidence
            label_text = f"Object {label}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame (optional)
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    # Write the annotated frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved to: {output_video_path}")
