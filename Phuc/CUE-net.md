---
id: CUE-net
aliases: []
tags: []
---

# Abstraction

- A novel architecture designed for automated violence detection in video surveillance.
- Is the combining version of:
  1. spatial Cropping
  2. UniformerV2 architecture (enhanced)
- Focusing on both local and global spatio-temporal features
- State-of-the-art performance on these datasets:
  1. RWF-2000
  2. RLVS

## 1. Introduction

1. CUE-Net, a novel architecture for violence detection video analytics which incorporates a novel enhanced version of the UniformerV2 architecture along with Modified Efficient Additive Attention (MEAA), a novel attention mechanism to capture the important global spatio-temporal features.
2. Incorporating a spatial cropping mechanism based on the detected number of people in our algorithm before the video is fed into the main learning algorithm, to focus the method on the area where violence is occurring without losing the important surrounding information.
3. Results set a new state-of-the-art on the RWF-2000 and RLVS datasets, outperforming the most recently published methods.

## 2. Related work

### 2.1 Anomaly Detection

### 2.2 In an Action Recognition Context

## 3. Proposed methods (CUE-Net)

- 5 main module:
  1. Spatial Cropping Module
  2. 3D Convolution Backbone
  3. Local UniBlock V2
  4. Global UniBlock V3
  5. Fusion Block

### 3.1 Spatial Cropping Module

### 3.2 3D Convolution Backbone

### 3.3 Local UniBlock V2

### 3.4 Global UniBlock V3

### 3.5 Fusion Block

## 4. Experiments and Results

## 5. Conclusion
