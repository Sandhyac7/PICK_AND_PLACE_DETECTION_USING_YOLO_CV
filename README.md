# Pick and Place Detection using YOLO and Computer Vision

## 📌 Project Overview

This project implements a computer vision system to detect and track a backpack in video footage and identify its pick-and-place state. It uses YOLO for object detection and OpenCV for video processing, Region of Interest (ROI) monitoring, and visualization.

The system analyzes interactions between a person and a backpack to determine whether the backpack has been picked up or placed down.

## 🎯 Objectives

- Detect backpacks in video frames using YOLO.
- Track objects and monitor their movement.
- Define a Region of Interest (ROI) for backpack interaction.
- Identify picking and placing activities based on detection and tracking information.
- Display the detected backpack, person ID, and current state on video frames.

## ✨ Key Features

- **YOLO-based detection:** Identifies backpacks in video footage.
- **Video processing:** Processes video frame by frame using OpenCV.
- **ROI monitoring:** Tracks interaction with a predefined region.
- **State classification:** Determines whether a backpack is picked up or placed down using rule-based logic.
- **Visual output:** Displays detection information and state labels on the processed video.

## 🛠️ Technologies Used

- Python
- Ultralytics YOLO
- OpenCV
- Computer Vision
- Object Tracking

## ⚙️ System Workflow

1. Load the YOLO model and input video.
2. Read video frames using OpenCV.
3. Detect the backpack and relevant person information.
4. Monitor the defined Region of Interest (ROI).
5. Analyze detection and tracking information to determine the backpack's state.
6. Display the detection results, person ID, and pick-and-place status.
7. Generate an annotated output video.

## 📂 Project Structure

```text
pick-and-place-detection-yolo/
├── input/
│   └── backpack_video.mp4
├── output/
│   └── result.mp4
├── main.py
├── yolo11n.pt
├── yolo11n-pose.pt
├── requirements.txt
└── README.md
```

*Note: Update this structure to match your actual files and script names. Model weights and videos may be excluded from the repository if they are large or unnecessary.*

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sandhyac7/PICK_AND_PLACE_DETECTION_USING_YOLO_CV.git
cd PICK_AND_PLACE_DETECTION_USING_YOLO_CV
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the required packages:

```bash
pip install ultralytics opencv-python
```

### 3. Configure the input

Place the input video and required YOLO model weights in the appropriate folders. Update the paths in the Python script if necessary.

### 4. Run the project

```bash
python main.py
```

Replace `main.py` with the actual name of your main Python script.

## 📊 Expected Output

The system produces an annotated video showing:

- Backpack detection results.
- Bounding boxes around detected objects.
- Person tracking information, where implemented.
- Picked or placed state labels.
- Processed video output for reviewing the detection results.

## 🔍 Challenges Addressed

- Temporary loss of backpack or wrist detection.
- Changes in object visibility and overlap with the ROI.
- Maintaining consistent tracking IDs.
- Distinguishing a temporary detection loss from a completed pick-and-place action.

## 🔮 Future Improvements

- Improve tracking consistency during occlusion.
- Enhance state classification under different camera angles and lighting conditions.
- Evaluate detection and state-classification performance.
- Extend the system to support multiple objects and more complex interactions.

## 👩‍💻 Author

**Sandhya C.**

Computer Science and Engineering

GitHub: [Sandhyac7](https://github.com/Sandhyac7)

---

*This project demonstrates the application of YOLO-based object detection and computer vision techniques for video-based pick-and-place activity analysis.*
