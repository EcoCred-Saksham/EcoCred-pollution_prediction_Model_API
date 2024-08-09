# EcoCred Model and Flask API

## Overview

"EcoCred" is a green credit management system that combines IoT sensors with blockchain technology to monitor pollution levels and allocate green credits to industries based on their environmental impact. This repository contains the machine learning model used to assess environmental data and the Flask API that interacts with the model for real-time monitoring and green credit management.

---

## Repository Structure

- **/model/**  
  Contains the machine learning model files, including the model architecture, training scripts, and saved weights.

- **/api/**  
  Houses the Flask API code responsible for handling requests, interacting with the model, and storing data on the Polygon blockchain.

- **/data/**  
  Sample datasets for testing and validating the model.

- **/docs/**  
  Documentation files explaining the project, including this README and other related documents.

- **/tests/**  
  Test cases for both the model and the Flask API to ensure functionality and accuracy.

---

## Prerequisites

- Python 3.8 or later
- Flask
- TensorFlow or PyTorch (depending on the model used)
- Web3.py (for blockchain interactions)
- Raspberry Pi setup (for IoT data collection)
- Polygon Blockchain account

---
