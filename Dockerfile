# Use an official Python runtime as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .


RUN python -m pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the rest of the files into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 22111

# Run the command to start the app
CMD ["python", "api.py"]
