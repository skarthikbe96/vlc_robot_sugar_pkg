# vlc_robot_sugar_pkg

ROS 2 package for visual–language–conditioned robotic perception and manipulation using 3D point clouds.  
This package is designed to integrate **SUGAR-style 3D visual representations** into ROS 2 pipelines for research and experimentation.

---

## Description

`vlc_robot_sugar_pkg` provides ROS 2 nodes, launch files, and configuration utilities for:
- 3D point cloud perception
- Language-conditioned object grounding
- Robotics-friendly 3D visual representations
- Integration with learning-based manipulation pipelines

The package is intended for **ROS 2 (colcon)** workflows and supports both simulation and real-robot setups depending on configuration.

---


##  Method Summary

- **Inputs**
  - Colored 3D point cloud
  - Natural-language prompt

- **Processing**
  - Extraction of semantic, geometric, and affordance-aware features
  - Grounding of objects in multi-object, cluttered 3D scenes
  - Learned representations aligned across vision, geometry, and language

- **Supported Capabilities**
  - 3D object recognition
  - Referring expression grounding
  - Language-guided robotic manipulation

- **System Design**
  - Modular ROS 2 nodes
  - Explicit separation of perception, grounding, and task interfaces
  - Designed for seamless **simulation ↔ real-world transfer**

---

## 📊 Datasets & Evaluation (Reference)

### Datasets
- **ShapeNet**  
  Single-object 3D assets for controlled evaluation
- **Objaverse**  
  Large-scale, diverse 3D object dataset
- **Synthetic multi-object scenes**  
  Generated in simulation for cluttered-scene grounding

### Evaluation Protocols
Downstream evaluations are inspired by:
- Zero-shot 3D object recognition
- Referring expression grounding
- Language-guided manipulation tasks

---

## 🧩 Use Cases

- Multimodal 3D perception research
- Language-conditioned robotic manipulation
- Simulation-to-real transfer experiments in ROS 2

---

## Acknowledgement

This work is inspired by the SUGAR framework for 3D visual representation learning in robotics.

```bibtex
@InProceedings{Chen_2024_SUGAR,
    author    = {Chen, Shizhe and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
    title     = {SUGAR: Pre-training 3D Visual Representations for Robotics},
    booktitle = {CVPR},
    year      = {2024}
}

```

## Requirements

- Ubuntu 24.0
- ROS 2 Jazzy 
- Python ≥ 3.8  
- colcon  
- PyTorch  
- Open3D  
- NumPy  
- OpenCV  

---

## Build (ROS 2)

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/skarthikbe96/vlc_robot_sugar_pkg.git
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Run SUGAR ROS 2 Nodes

```bash
ros2 launch vlc_robot_sugar_pkg sugar_pipeline.launch.py
```

## Publish Language / Object Prompt

```bash
ros2 topic pub -1 --qos-durability transient_local \
/vlc/text_prompt std_msgs/msg/String "{data: 'red mug'}"
```

