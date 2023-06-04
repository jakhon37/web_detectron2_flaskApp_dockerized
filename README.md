# Web Detectron2 Flask App Dockerized

This repository contains a Dockerized Flask application that uses the Detectron2 library for object detection tasks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

```bash
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2

RUN apt-get update && apt-get install -y libgl1-mesa-glx
