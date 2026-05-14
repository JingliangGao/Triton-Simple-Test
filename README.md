# Triton-Simple-Test
Simple test case for Nvidia Triton 

## Prepare
1. create Python environment
```bash
python3 -m venv llm_env
source llm_env/bin/activate
```

2. install python package
```bash
pip3 install -r requirements.txt
```

## Run case
```bash
python3 llm_test.py -m <your-deployed-model>  # llm_Qwen-2.5-3b_1.0
```