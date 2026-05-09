#!/bin/bash

# set variables
CURRENT_DIR=$(pwd)

# install python dependencies
pip3 install -r requirements.txt

# install system dependencies from Kylin repository
sudo apt install -y llamacpp llm-backend

# run case
cd ${CURRENT_DIR}

python3 llm_unit_test.py -m llm_Qwen-2.5-3b_1.0