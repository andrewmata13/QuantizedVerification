#!/bin/bash

TOOL="iq-verify"
CATEGORY="ainncs"
IMAGE_NAME="wehbedoug/iqverify:20250606"
CONTAINER_NAME="${TOOL}-container-$1"

# Setup
mkdir ./results

# Build the image
docker build . -t "$IMAGE_NAME" --no-cache


# Run the benchmarks
docker run --name "$CONTAINER_NAME" "$IMAGE_NAME" bash -c "python examples/robot_navigation/robot.py --q_x=0.1 --q_y=0.1 --q_nu=0.1 --q_theta=0.1 --q_u1=0.1 --q_u2=0.1 --show_plot=False --use_robust_nn=True"


# Download results from container to local machine
docker cp "$CONTAINER_NAME":/home/results.csv   ./results/results.csv


# Clean up container and image
docker rm --force "$CONTAINER_NAME"
docker image rm --force "$IMAGE_NAME"
