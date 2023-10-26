# WALI-HRI: Worker Attention Lapse for Industrial HRI

### Introduction

This dataset aims to foster affective computing in human attention, aka engagement, 
tracking in industrial Human-Robot Interaction settings. A multimodal sensing of 
manifold worker attention lapses are recorded and timestamp annotated in order to 
endow perception-cognition-action capabilities for robotic and autonomous agents.

For details, please read and cite our paper:

```
@ARTICLE{dai_wali2023,
  author={Dai, Zhuangzhuang and XX},
  journal={RA-L}, 
  title={WALI-HRI}, 
  year={2023},
  volume={1},
  number={},
  pages={1-2}}
```

------

### Data Sample Visualization

![openingfig](https://github.com/zdai257/WALI-HRI/blob/main/imgs/AudioVideoView3.png)

------

### Pre-requisites

Sensors and associated data formats are as follows

**Sensor** | **Feature** | **Data File**
--- | --- | ---
Azure Top View | Visible, gesture, objects | `YUV.mp4`, `Depth.mp4`
Azure 3rd Person View | Posture, head pose, facial expression, Infrared | `Depth.mp4`, `IR.mp4`
Eye Tracking | Gaze, Fixation, Blinks, Head pose, Surface, annotated disengagement | `world.mp4`, annotation.csv, fixation.csv, blinks.csv, head_pose_track.csv
Auditory recording | Conversation, background sound | `sound_wave.wav`

### Usage

To download the dataset

```bash
./downloader.sh
```

The dataset contains multimodal (visual, depth, infrared, eye tracking, and auditory) sensing of human affects grouped by human participant's index. To load the dataset as a python dictionary

```python
import preprocess

data = preprocess.load()
```

### Data Collection Pipeline

Sychronisation of multimodal data streams is implemented [here](https://github.com/Junaid0411/AstonAttentionLapseResearchProject). Ensure Azure Kinect DK camera(s) and Pupil Capture for eye tracker are connected and installed properly. Use *utils/recordGUI.py* or command below to launch the master recorder

```bash
python utils/record_capture.py
```

Use command below to launch recording of any slave Azure Kinect DK camera before launching the master

```bash
k4arecorder --external-sync sub -l 300 /home/path/repo/Azure-Kinect-Sensor-SDK/dataset/2023_XX_XX/ROBLAB_0X/sub1.mkv
```

UR Script for Universal Robots the pick-and-handover routine is provided in *UR-program* folder.

------

### Acknowledgement

This work was funded by 2022/23 Aston Pump Priming Scheme and AAU Bridging Project 
"A Multimodal Attention Tracking In Human-robot Collaboration For Manufacturing 
Tasks."

We thank the Greater Birmingham and Solihull Institute of Technology (GBSIoT) and the 
Aalborg 5G Smart Production Lab for supporting our data collection campaign.

