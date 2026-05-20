#!/bin/bash

# set variables
CURRENT_DIR=$(pwd)

# install system dependencies from Kylin repository
# sudo apt install -y llamacpp llm-backend
sudo apt install -y python3-venv

# create and activate 'llm_env' environment
echo "[INFO] Create and activate 'llm_env' environment ... "
python3 -m venv llm_env
source llm_env/bin/activate

# install python dependencies
echo "[INFO] Start to install dependencies ... "
pip3 install -r requirements.txt 

# run case
echo "[INFO] Start to run case ... "
cd ${CURRENT_DIR}
python3 llm_test.py -m llm_Qwen-2.5-3b_1.0

echo "[INFO] All done. "

                    