# Object Detection and Tracking with Unique ID Assignment
This project implements object detection and tracking using the TensorFlow pre-trained model ssd_mobilenet_v2. The goal is to detect and assign unique IDs to individuals in video footage, saving their images and tracking their appearance throughout multiple frames.

# Project Structure
TensorFlow Model: We use the pre-trained ssd_mobilenet_v2 from TensorFlow Hub for person detection.
Video Processing: Detects objects (persons) in videos, assigns unique IDs to each individual, and stores detected person images.
CSV Reporting: Outputs a CSV file summarizing the person ID, the frame number, video number, and the number of appearances per person.
# Files & Directories:
videos/: Contains the videos to be processed.
frames/: Stores frames of detected persons, organized by video.
CSV_Outputs/: Stores CSV files for each video summarizing the results.
README.md: Project documentation (this file).
object_detection.py: Main Python script for object detection and tracking.
# Setup
Clone the repository or download the project files.
Upload the video files to your Google Drive (make sure they are in .mp4 format).
Run the project in Google Colab to process the videos and generate the output.
# Required Libraries:
OpenCV
TensorFlow
TensorFlow Hub
NumPy
CSV
You can install the required libraries using the following commands:

!pip install opencv-python tensorflow numpy tensorflow-hub
# Google Drive Setup
Make sure to mount your Google Drive in Colab before running the project:

from google.colab import drive
drive.mount('/content/drive')
# How to Run
Upload Videos: Place your videos in a directory on your Google Drive.
Model Loading: The TensorFlow model is automatically downloaded from TensorFlow Hub.
Video Processing: The script processes all videos, extracts frames containing persons, and assigns unique IDs.
CSV Generation: For each video, a CSV file is generated, summarizing the number of appearances per detected person.
# Steps in Colab:
Mount Google Drive.
Update paths for videos_dir, frames_dir, and output_csv_path to the correct Google Drive directories.
Run the script to perform detection, tracking, and save results.
# Sample Usage:
Set paths to Google Drive directories
videos_dir = '/content/drive/MyDrive/ObjectDetection/Videos'
frames_dir = '/content/drive/MyDrive/ObjectDetection/Frames'
output_csv_path = '/content/drive/MyDrive/ObjectDetection/CSV_Outputs/detections_summary.csv'

# Load the model
model = load_model()

# Process videos and save CSV
all_video_data = process_all_videos(videos_dir, frames_dir, model)
save_to_csv(all_video_data, output_csv_path)
# Output
Frames Directory: Stores cropped images of detected persons, organized by video and frame number.
CSV File: Contains a summary of each person's appearance across frames, including their assigned ID and total count.
# Future Enhancements
Implement re-identification algorithms to track individuals across multiple videos.
Add post-occlusion tracking to improve ID consistency.
Further optimize processing efficiency for longer videos.
