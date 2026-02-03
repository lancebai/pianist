# Base image for NVIDIA Jetson (L4T - Linux for Tegra)
# This base image includes CUDA, cuDNN, and TensorRT pre-installed.
# CHECK COMPATIBILITY: r35.2.1 is for JetPack 5.1. Update tag based on your actual JetPack version.
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# 1. Install System Dependencies
# - libopencv-dev & python3-opencv: Installs OpenCV optimized for Tegra (with GStreamer support)
# - libgstreamer1.0-dev: Required for video pipelines
# - v4l-utils: For camera controls (v4l2-ctl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    libgstreamer-plugins-base1.0-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Working Directory
WORKDIR /app

# 3. Install Python Requirements
# Note: We rely on the system-pacakge 'python3-opencv' for hardware acceleration support.
# Do NOT install 'opencv-python' via pip, as it overrides the Tegra-optimized version.
# We create a requirements.txt excluding opencv.
COPY deploy/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 4. Copy Application Code
COPY . .

# 5. Runtime Configuration
# - Allow access to video devices
# - Expose Telemetry Port
EXPOSE 5000

# 6. Entrypoint
# We use 'python3' directly.
# NOTE: To run this container, you MUST use '--runtime nvidia' and '--device /dev/video0'
CMD ["python3", "main.py", "--input_type", "camera", "--input_path", "0", "--no_display"]
