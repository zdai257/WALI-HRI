# WALI-HRI: Worker Attention Lapse for Industrial HRI

### Introduction

This repository presents WALI-HRI dataset, with fully annotated 
disengagement/distraction events, collected in an industrial phone assembly scenario 
of human-robot collaboration. Multimodal sensing streams of 28 
human participant's attention tracking are made publicly available. The aim is to 
endow perception and cognitive safety capabilities for intellgent robots and 
autonomous systems.

For details, please read and cite our paper:

```
@ARTICLE{dai_walihri2024,
  author={Dai, Zhuangzhuang and Park, Jinha and Akhtar, Junaid and Kaszowska, 
Aleksandra and Li, Chen},
  journal={arXiv}, 
  title={WALI-HRI}, 
  year={2024},
  volume={1},
  number={},
  pages={1-7}}
```

------

### Data Sample Visualization

![openingfig](https://github.com/zdai257/WALI-HRI/blob/main/imgs/AudioVideoView3.png)

Sensor modalities and associated data formats are as follows

**Sensor** | **Feature** | **Data File**
--- | --- | ---
Azure Top View | Depth, Infrared, Gesture, Objects | `Depth.mp4`, `IR.mp4`
Azure 3rd Person View | Depth, Infrared, Posture, Head Pose, Facial Expression | `Depth.mp4`, `IR.mp4`
Eye Tracking | Gaze, Fixations, Blinks, Head pose, Surface Tracking, Disengagement Events, and raw data | `world.mp4`, *fixation.csv*, *blinks.csv*, *head_pose_tracker_poses.csv*, *surface_events*, *annotation.csv*
Auditory Recording | Speech, Ambient Sound | `sound_recording.wav`

------

### Pre-requisites

To reproduce synchronised recorder and environment for running Pupil eye tracker

```bash
# Install System Requirements with Conda
conda create --name <env_name> --file requirements.txt
```

If only to use the dataset, you do not need to recreate the environment.

### Usage

The WALI-HAI dataset is publicly available at a [**Box shared drive**](https://aston.box.com/s/vdh27hmczaoyx4t8rbsobjf6mjn67kmb). Note the full dataset size is about 105GB.

To download a data sample

```bash
./downloader.sh
```

The dataset contains multimodal (visual, depth, infrared, eye tracking, and auditory) sensing of human affects grouped by anonymized and randomized participant's index. *Annotations.xlsx* contains labelled disengagement events for all data samples.

To load the dataset as a *pandas DataFrame* and resample the multimodal data in a synchronised fasion

```base
python3 preprocess.py --type csv
```

or directly load the pre-processed *data_pkl.pkl* pickle file. Note you need *python=3.11* and *pandas=1.5.3* to load the pickle file directly.

### Data Collection Pipeline

Real-time collection of multimodal data is implemented [here](https://github.com/Junaid0411/AstonAttentionLapseResearchProject). Ensure Azure Kinect DK camera(s) and Pupil Capture for eye tracker are connected and installed properly. Use *utils/recordGUI.py* or command below to launch the master recorder

```bash
python3 utils/record_capture.py
```

Use command below to launch recording of any slave Azure Kinect DK camera before launching the master

```bash
k4arecorder --external-sync sub -l 300 /home/path/repo/Azure-Kinect-Sensor-SDK/dataset/2023_XX_XX/ROBLAB_0X/sub1.mkv
```

UR Script for controlling Universal Robots (tested on UR3 and UR5) pick and handover routine is provided in *UR-program* folder.

------

### License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is available under the MIT License. Contributions are not accepted. If you notice issues, feel free to raise them or email the authors.

------

### Acknowledgement

This work was funded by 2022/23 Aston Pump Priming Scheme and AAU Bridging Project 
"A Multimodal Attention Tracking In Human-robot Collaboration For Manufacturing 
Tasks."

We thank the Greater Birmingham and Solihull Institute of Technology (GBSIoT) and the 
Aalborg 5G Smart Production Lab for supporting the data collection campaign.
