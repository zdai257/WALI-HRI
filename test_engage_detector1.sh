#!/bin/bash

# Define the root folder to search for sub-folders
root_folder="/home/CAMPUS/daiz1/repo/WALI-HRI/dataset"

# Check if the user has provided arguments of the target dataset path
if [ "$#" -ge 1 ]; then
  root_folder="$1"
fi

# Extract the roslaunch arguments and rosbag record arguments
dirpath="$1"

# Use find to locate all sub-folders within the root folder
subfolders=$(find "$root_folder" -mindepth 1 -maxdepth 1 -type d)

echo "Launch engage detector"
(cd "$root_folder" && roslaunch engagement_detector engagement_detector.launch) &

sleep 15

# Loop through the sub-folders
for subfolder in $subfolders; do
  echo "Processing sub-folder: $subfolder"
  
  file_path="$subfolder/Azure_3rd/YUV.mp4"
  echo "Tested file path: $file_path"

  # Execute the engagement detector command
  echo "Supply visible video recording: $file_path"
  (roslaunch video_stream_opencv video_file.launch vidpath:="$file_path") &

  # Recording values
  subfolder_name=$(basename "$subfolder")
  bag_path="/home/CAMPUS/daiz1/Documents/LCAS_bags/$subfolder_name.bag"
  echo "Recording to: $bag_path"
  (rosbag record -O "$bag_path" --duration=305 /engagement_detector/value)

  # Sleep between runs (optional, adjust as needed)
  sleep 5

  # You may also want to add error handling or logging
done

