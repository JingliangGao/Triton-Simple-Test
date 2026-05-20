# Triton-Simple-Test
Nvidia Triton 推理服务框架简易测试脚本

English | [简体中文](README_zh.md)

## 环境配置
1. 构建Python虚拟环境
```bash
python3 -m venv llm_env
source llm_env/bin/activate
```

2. 安装Python依赖包
```bash
pip3 install -r requirements.txt
```

## 运行案例
查询部署模型的名称，即`ls /opt/appdata/kylin-ai/model-repository/`. 测试模型，执行命令：
```bash
python3 llm_test.py -m <your-deployed-model>  # llm_Qwen-2.5-3b_1.0
```