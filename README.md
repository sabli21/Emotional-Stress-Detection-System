# Stress Detection System

A real-time stress detection system using computer vision and machine learning, capable of analyzing heart rate, facial emotions, and facial feature movements.

## Table of Contents
- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)

## Introduction
Stress can negatively affect a person’s performance and health. This system integrates various modules to detect stress levels using real-time video input.

## System Architecture
1. **Facial Emotion Recognition**: Identifies emotions like anger, sadness, and fear.
2. **Heart Rate Detection**: Uses photoplethysmography to calculate heart rate.
3. **Facial Feature Extraction**: Tracks eye blinks, eyebrow displacement, and lip movements.

## Features
- **Real-time Processing**: Detects stress in live video streams.
- **Web Interface**: User-friendly interface for stress assessment.
- **Integration of Multiple Features**: Combines physiological and behavioral indicators.

## Requirements
- Python 3.8+
- Libraries: OpenCV, TensorFlow, Flask, Numpy, Pandas, Dlib, Scipy, Matplotlib
- Files: Pre-trained models (`first_5322_model.hdf5`, `my_model.pkl`), datasets (`total_input.txt`, `total_label_v2.txt`)

Install dependencies:
```bash
pip install -r requirements.txt

##Setup
##Clone the repository:
git clone <repository_url>
cd Stress-Detection-System

##Run the Flask application:
python app.py

##Open the web interface at http://localhost:5000.

Usage
Upload a video or use a live webcam feed.
View stress level predictions based on heart rate, facial emotions, and features.
Results
Module Accuracies:
Emotion Recognition: 80.61%
Heart Rate Detection: 89%
Stress Detection (integrated): 75%
Example output:

Future Work
Enhance heart rate detection for low-light conditions.
Expand dataset for better accuracy.
Include more physiological indicators (e.g., skin conductance).