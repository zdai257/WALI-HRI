# WALI-HRI: Worker Attention Lapse for Industrial HRI

### Introduction

This repository presents [WALI-HRI dataset](https://www.youtube.com/watch?v=tGt8HuLLyI0), with fully annotated 
disengagement/distraction events, collected in an industrial phone assembly scenario 
of human-robot collaboration. Multimodal sensing streams of 28 
human participant's attention tracking are made publicly available. The aim is to 
endow perception and cognitive safety capabilities for intelligent robots and 
autonomous systems.

For details, please read and cite our paper:

```
@ARTICLE{dai_walihri2024,
  author={Dai, Zhuangzhuang and Park, Jinha and Akhtar, Junaid and Kaszowska, 
Aleksandra and Li, Chen},
  journal={arXiv}, 
  title={WALI-HRI}, 
  year={2024},
  volume={},
  number={},
  pages={1-8}}
```

AND

```
@INPROCEEDINGS{dai_icac2023,
  author={Dai, Zhuangzhuang and Park, Jinha and Kaszowska, Aleksandra and Li, Chen},
  booktitle={IEEE 28th International Conference on Automation and Computing (ICAC)}, 
  title={Detecting Worker Attention Lapses in Human-Robot Interaction: An Eye Tracking and Multimodal Sensing Study}, 
  year={2023},
  volume={},
  number={},
  pages={558-563},
  doi={10.1109/icac57885.2023.10275177}}
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

To reproduce the dataset pre-processor, data loader, and training pipeline

```bash
# Install System Requirements with Conda
conda create --name <env_name> --file requirements.txt
```

If only to download the dataset, you do not need to recreate the environment.

### Usage

The WALI-HAI dataset is publicly available at a [**Box shared drive**](https://aston.box.com/s/vdh27hmczaoyx4t8rbsobjf6mjn67kmb). Note the full dataset size is about 105GB.

To download one data sample,

```bash
./downloader.sh
```

The dataset contains multimodal (visual, depth, infrared, eye tracking, and auditory) sensing of human affects grouped by anonymized and randomized participant's index. *Annotations.xlsx* contains labelled disengagement events for all data samples.

To load the dataset as a *pandas DataFrame* and resample the multimodal data in a synchronised fashion

```base
python3 preprocess.py --type csv
```

or directly load the pre-processed *data_pkl.pkl* pickle file. Note it requires *python=3.11* and *pandas=1.5.3* to load the pickle file directly.

Neural network training requires *pytorch=1.13.1* with hyperparams stored in *configuration.yaml*. Use command below to start training,

```base
python3 train.py
```

and the following for evaluation and visualization of results,

```base
python3 Testing.py
```

------

### Data Collection Pipeline

Runtime data collection pipeline of multimodal data is implemented [here](https://github.com/Junaid0411/AstonAttentionLapseResearchProject). Ensure Azure Kinect DK camera(s) and Pupil Capture for eye tracker are connected and installed properly. Use *utils/recordGUI.py* or command below to launch the master recorder

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

This work was funded by 2022/23 Aston Pump Priming Scheme, and partially supported by AI for the People and Aalborg Robotics Challenge (ARC) Bridging Project, and by Innovation Fund Denmark as a part of the project “MADE FAST”. 

We thank the Greater Birmingham and Solihull Institute of Technology (GBSIoT) and the Aalborg 5G Smart Production Lab for supporting the data collection campaign.
