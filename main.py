import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
video_path = 'C:/Users/sagor/PycharmProjects/trafficProject/Red Light Violation.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define initial points to form the ROI (Region of Interest)
# These points need to be adjusted after visually inspecting the frames
roi_vertices = np.array([[(100, 200), (100, 600), (600, 600), (600, 200)]], dtype=np.int32)

def roi_mask(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked


# Read a frame from the video
ret, frame = cap.read()
if not ret:
    print("Failed to grab a frame from the video.")
    cap.release()
    exit()

# Apply the ROI mask to the frame
masked_frame = roi_mask(frame, roi_vertices)

# Plot the original and masked frame to adjust the ROI
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Original Frame")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB))
plt.title("Frame with ROI")

plt.show()


# Updated ROI vertices after manual inspection
roi_vertices = np.array([[(50, 250), (50, 500), (700, 500), (700, 250)]], dtype=np.int32)




while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update the light_state and vehicle detection logic as needed
    light_state = detect_traffic_light(frame)
    processed_frame, violations = process_frame(frame, background_frame, light_state)

    # Draw the ROI on the frame
    cv2.polylines(processed_frame, [roi_vertices], isClosed=True, color=(255, 255, 0), thickness=5)

    # Display the frame
    cv2.imshow('Processed Frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

