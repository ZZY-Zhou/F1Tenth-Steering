# Steering Prediction via a Multi-Sensor System for Autonomous Racing (ICRA'25)

This repository is for the paper **Steering Prediction via a Multi-Sensor System for Autonomous Racing**, by
[Zhuyun Zhou](https://scholar.google.com/citations?user=sXolUXMAAAAJ&hl=en&oi=ao),
[Zongwei Wu](https://scholar.google.com/citations?user=3QSALjX498QC&hl=en&oi=ao),
Florian Bolli,
[Rémi Boutteau](https://scholar.google.com/citations?user=U-SrcPkAAAAJ&hl=en&oi=ao),
[Fan Yang](https://scholar.google.com/citations?user=GNQHje8AAAAJ&hl=en&oi=ao),
[Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en&oi=ao),
[Dominique Ginhac](https://scholar.google.com/citations?user=fkdCT5kAAAAJ&hl=en&oi=ao),
[Tobi Delbruck](https://scholar.google.com/citations?user=hnl-RQQAAAAJ&hl=en&oi=ao).

PDF version of the paper is available [here](https://arxiv.org/pdf/2409.19356).

Dataset ***EveLidAngle*** can be found [here](#dataset).



## Contents

1. [Abstract](#abstract)
2. [Dataset](#dataset)
3. [Citation](#citation)
4. [Installation](#installation)
5. [Training](#training)
6. [Testing](#testing)
7. [Acknowledgment](#acknowledgment)


## Abstract

![Graphical Abstract](https://github.com/ZZY-Zhou/F1Tenth-Steering/blob/main/Graphical%20Abstract%20ICRA'25.png)


Autonomous racing has rapidly gained research attention. Traditionally, racing cars rely on 2D LiDAR as their primary visual system. In this work, we explore the integration of an event camera with the existing system to provide enhanced temporal information. Our goal is to fuse the 2D LiDAR data with event data in an end-to-end learning framework for steering prediction, which is crucial for autonomous racing. To the best of our knowledge, this is the first study addressing this challenging research topic. We start by creating a multisensor dataset specifically for steering prediction. Using this dataset, we establish a benchmark by evaluating various SOTA fusion methods. Our observations reveal that existing methods often incur substantial computational costs. To address this, we apply low-rank techniques to propose a novel, efficient, and effective fusion design. We introduce a new fusion learning policy to guide the fusion process, enhancing robustness against misalignment. 
Our fusion architecture provides better steering prediction than LiDAR alone, significantly reducing the RMSE from 7.72 to 1.28. Compared to the second-best fusion method, our work represents only 11\% of the learnable parameters while achieving better accuracy.



## Dataset

***EveLidAngle*** can be downloaded [here](https://drive.google.com/file/d/1Bm8yLOnflttyp_G9F4kwzO-0AuzmZX0w/view?usp=drive_link).

In total, our EveLidAngle dataset contains 7 sequences / ROS bags (27452 valid event-LiDAR-steering angle pairs), with 5 sequences / ROS bags (21576 valid event-LiDAR-steering angle pairs) for training and 2 other sequences / ROS bags (5876 valid event-LiDAR-steering angle pairs) for testing.

In each sequence:
* `lidar_depth`: lidar scans with depth information;
* `event_frame`: event frames synchronised to lidar scans, so that event frames and lidar scans are corresponding;
* `steering_angles`: ground truth steering angles.


The format should be:
```
└── EveLidAngle
    ├── training
    │   ├── forward_11
    │   │   ├── lidar_depth
    │   │   │   ├── xxxxxx.png
    │   │   │   └── ...
    │   │   ├── event_frame
    │   │   │   ├── xxxxxx.png
    │   │   │   └── ...
    │   │   └── steering_angles.txt  
    │   └── ...
    └── testing
        ├── forward_21
        │   └── ...
        └── ... 
```



## Citation

```BibTeX
@article{zhou2024steering,
  title={Steering prediction via a multi-sensor system for autonomous racing},
  author={Zhou, Zhuyun and Wu, Zongwei and Bolli, Florian and Boutteau, R{\'e}mi and Yang, Fan and Timofte, Radu and Ginhac, Dominique and Delbruck, Tobi},
  journal={arXiv preprint arXiv:2409.19356},
  year={2024}
}
```





## Installation

1. Clone

```
git clone https://github.com/ZZY-Zhou/F1Tenth-Steering
cd F1Tenth-Steering
```

2. Create and activate conda environment

```
conda create -n ENV_NAME
conda activate ENV_NAME
```



## Training

```
python3 train.py --train_data PATH_TO_DATASET/EveLidAngle/training --test_data PATH_TO_DATASET/EveLidAngle/testing
```



## Testing



```
python3 metrics.py output.txt
```



## Acknowledgment

The authors sincerely thank Liam Boyle, Nicolas Baumann, Jonas Kühne, and Niklas Bastuck from PBL, ETH Zürich, for their support with recordings and expertise on the F1tenth car, which are essential for this work.
