# Worker Attention Lapse for Industrial HRI

### Introduction

This dataset aims to foster affective computing in human attention, aka engagement, 
tracking in industrial Human-Robot Interaction settings. A multimodal sensing of 
manifold worker attention lapses are recorded and timestamp annotated in order to 
endow perception-cognition-action capabilities for robotic and autonomous agents.

For details, please check our paper:

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

### Download and Pre-processing

To download

```bash
git clone https://github.com/zdai257/WALI-HRI.git
```

To reproduce data analytics

```bash
python3 src/display.py
```

### Data Collection Steps

Use command below to launch synchronised recording of slave Azure Kinect

```bash
k4arecorder --external-sync sub -l 300 /home/path/repo/Azure-Kinect-Sensor-SDK/dataset/2023_XX_XX/ROBLAB_0X/sub1.mkv
```

------

### Acknowledgement

This work was funded by 2022/23 Aston Pump Priming Scheme and AAU Bridging Project 
"A Multimodal Attention Tracking In Human-robot Collaboration For Manufacturing 
Tasks."

We thank the Greater Birmingham and Solihull Institute of Technology (GBSIoT) and the 
Aalborg 5G Smart Production Lab for supporting our data collection campaign.

