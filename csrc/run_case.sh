#!/bin/bash

# set variables
CURRENT_DIR=$(pwd)
BUILD_FOLDER="build-debug"

# install system dependencies 
cd ${CURRENT_DIR}
sudo apt install -y catch2 tritonclient

# refresh build directory
cd ${CURRENT_DIR}
if [ -d "$BUILD_FOLDER" ]; then
    echo "Build directory already exists. Removing it..."
    rm -rf "$BUILD_FOLDER"
fi
mkdir "$BUILD_FOLDER"

# build
cd ${CURRENT_DIR}/${BUILD_FOLDER}
cmake ../
make -j$(nproc)

