#!/usr/bin/env python
'''
Copyright 2024. All Rights Reserved.
Author: Arthur Bin
Date: 2024-09-27 15:27:13
Description: No streaming infer, optimized version
'''
import sys
import argparse
import json
import queue
import time
from enum import Enum
from functools import partial
from typing import Optional, Tuple, Dict, Any

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

try:
    import tritonclient.http as httpclient
    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    HTTP_CLIENT_AVAILABLE = False


# ============================================================================
# 类型转换函数
# ============================================================================

def np_to_triton_dtype(np_dtype) -> Optional[str]:
    """将 NumPy 数据类型转换为 Triton 数据类型"""
    type_map = {
        bool: "BOOL",
        np.int8: "INT8",
        np.int16: "INT16",
        np.int32: "INT32",
        np.int64: "INT64",
        np.uint8: "UINT8",
        np.uint16: "UINT16",
        np.uint32: "UINT32",
        np.uint64: "UINT64",
        np.float16: "FP16",
        np.float32: "FP32",
        np.float64: "FP64",
    }
    if np_dtype in type_map:
        return type_map[np_dtype]
    if np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return "BYTES"
    return None


def prepare_tensor(name: str, data: Any, shape: list, dtype: type) -> grpcclient.InferInput:
    """创建 Triton 推理输入张量"""
    tensor_data = np.array(data, dtype=dtype) if isinstance(data, (list, np.ndarray)) else np.array([data], dtype=dtype)
    tensor = grpcclient.InferInput(name, shape, np_to_triton_dtype(dtype))
    tensor.set_data_from_numpy(tensor_data)
    return tensor


# ============================================================================
# 请求类型定义
# ============================================================================

class RequestType(Enum):
    COMPLETIONS = 0
    OPENAI_COMPLETIONS = 1
    OPENAI_CHAT = 2
    TOKENIZE = 3
    DETOKENIZE = 4
    #LORA_SET = 5
    LORA_GET = 6
    TASK_CANCEL = 7
    TASK_ABORT = 8
    METRICS = 9
    WORKER_SAVE = 10
    WORKER_RESTORE = 11
    WORKER_ERASE = 12


# ============================================================================
# 配置和常量
# ============================================================================

DEFAULT_CONFIG = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "n_keep": 0,
    "n_predict": -1,
    "cache_prompt": True,
    "stop": "",
    "stream": True,
    "repeat_penalty": 1.0,
    "lora_scale": [],
}

COMPLETION_REQUEST = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "n_keep": 0,
    "n_predict": 10,
    "cache_prompt": True,
    "stop": "<|im_end|>",
    "stream": True,
    "repeat_penalty": 1.1,
    "lora": [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}],
    "prompt": "你是谁？"
}

CHAT_REQUEST = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "n_keep": 0,
    "n_predict": 10,
    "cache_prompt": True,
    "stop": "<|im_end|>",
    "stream": True,
    "repeat_penalty": 1.1,
    "lora": [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}],
    "messages": [
        {"role": "system", "content": "你是个聊天助手。"},
        {"role": "user", "content": "你是谁？"}
    ]
}

DEFAULT_REQUEST_ID = "chat-1"


# ============================================================================
# 辅助类和函数
# ============================================================================

class UserData:
    """用户数据类，用于处理异步结果"""
    def __init__(self):
        self._completed_requests = queue.Queue()


def _callback(user_data: UserData, result: grpcclient.InferResult, error: Optional[InferenceServerException]):
    """回调函数，用于收集异步结果"""
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def prepare_inputs(
    request_type: int,
    request_id: str,
    text_input: str,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.95,
    n_keep: int = 0,
    n_predict: int = -1,
    cache_prompt: bool = True,
    stop: str = "",
    stream: bool = True,
    lora_scale: list = None,
    repeat_penalty: float = 1.0
) -> list:
    """准备推理输入"""
    lora_scale = lora_scale or []
    return [
        prepare_tensor("request_type", request_type, [1], np.int32),
        prepare_tensor("request_id", request_id, [1], np.object_),
        prepare_tensor("text_input", text_input, [1], np.object_),
        prepare_tensor("temperature_input", temperature, [1], np.float32),
        prepare_tensor("top_k_input", top_k, [1], np.int32),
        prepare_tensor("top_p_input", top_p, [1], np.float32),
        prepare_tensor("n_keep_input", n_keep, [1], np.int32),
        prepare_tensor("n_predict_input", n_predict, [1], np.int32),
        prepare_tensor("cache_prompt_input", cache_prompt, [1], bool),
        prepare_tensor("stop_input", stop, [1], np.object_),
        prepare_tensor("stream_input", stream, [1], bool),
        prepare_tensor("lora_scale_input", lora_scale, [len(lora_scale)], np.float32),
        prepare_tensor("repeat_penalty_input", repeat_penalty, [1], np.float32),
    ]


def send_request(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    user_data: UserData,
    request_type: int,
    **kwargs
) -> bool:
    """
    发送请求并等待结果
    
    返回:
        success
    """
    inputs = prepare_inputs(
        request_type=request_type,
        request_id=kwargs.get('request_id', DEFAULT_REQUEST_ID),
        text_input=kwargs.get('text_input', ''),
        temperature=kwargs.get('temperature', DEFAULT_CONFIG['temperature']),
        top_k=kwargs.get('top_k', DEFAULT_CONFIG['top_k']),
        top_p=kwargs.get('top_p', DEFAULT_CONFIG['top_p']),
        n_keep=kwargs.get('n_keep', DEFAULT_CONFIG['n_keep']),
        n_predict=kwargs.get('n_predict', DEFAULT_CONFIG['n_predict']),
        cache_prompt=kwargs.get('cache_prompt', DEFAULT_CONFIG['cache_prompt']),
        stop=kwargs.get('stop', DEFAULT_CONFIG['stop']),
        stream=kwargs.get('stream', DEFAULT_CONFIG['stream']),
        lora_scale=kwargs.get('lora_scale', DEFAULT_CONFIG['lora_scale']),
        repeat_penalty=kwargs.get('repeat_penalty', DEFAULT_CONFIG['repeat_penalty'])
    )
    
    client.async_stream_infer(model_name, inputs)

    answer = ""
    metrics = {}
    
    while True:
        data_item = user_data._completed_requests.get()
        if isinstance(data_item, InferenceServerException):
            return False
        
        final_response = data_item.get_response().parameters.get('triton_final_response', {}).bool_param
        output_data = data_item.as_numpy("text_output")
        if output_data is not None and len(output_data) > 0:
            out_str = output_data[0][0].decode('utf-8')
            print(out_str, end='', flush=True)
            answer += out_str
        
        output_token_data = data_item.as_numpy("token_output")
        if output_token_data is not None:
            metrics['tokens'] = output_token_data
        
        metrics_output = data_item.as_numpy("metrics_output")
        if metrics_output is not None:
            metrics['values'] = metrics_output
        
        if final_response:
            break
    
    print()
    return True


def test_completions_with_cancel(
    triton_client: grpcclient.InferenceServerClient,
    model_name: str,
    user_data: UserData,
    test_generation_cancel: bool = True
) -> None:
    """测试 completions 请求和取消功能"""
    def _completions_thread():
        print("\nTest: completions request (thread)")
        if test_generation_cancel:
            send_request(triton_client, model_name, user_data, RequestType.COMPLETIONS.value, 
                            text_input= "你是谁", 
                            n_predict=100)
        else:
            send_request(triton_client, model_name, user_data, RequestType.COMPLETIONS.value, 
                            text_input= (
                                "21世纪以来，人工智能（Artificial Intelligence, AI）技术以惊人的速度发展，从实验室走向现实，深刻改变了人类社会的生产方式、生活方式和思维模式。"
                                "从自动驾驶汽车到医疗诊断系统，从智能语音助手到工业机器人，AI的应用场景不断扩展，甚至开始挑战人类在创造力、决策力等领域的边界。这场由数据驱动的技术革命，"
                                "不仅推动了经济结构的升级，也引发了关于伦理、就业、安全等领域的广泛讨论。站在2025年的时间节点回望，AI智能体早已超越工具属性，"
                                "北邮刘伟教授描绘的“超个体智能”图景正在显现：当健康监测、日程管理、知识获取等智能体深度融入生活，人类认知边界被指数级拓展。"
                                "在贵州毕节，返乡青年通过AI直播矩阵卖出230万斤猕猴桃；在上海陆家嘴，白领的“私人律师”智能体3分钟生成合规租房合同，技术普惠的光芒照亮每个角落。"
                                "这场变革远未抵达终章。百度提出的“可插拔智能生态”、蚂蚁集团“有手有脚能办事”的蓝图，预示着更激动人心的未来：当智能体集群接入国家电网调度系统，"
                                "风光储协同效率将提升60%；当气候预测智能体与碳交易市场联通，或可提前三年实现碳中和目标；当药物研发智能体与脑机接口结合，"
                                "阿尔茨海默病的治疗可能迎来突破……正如OpenAI CEO山姆·奥尔特曼在2025世界人工智能大会上所言：“AI不是在替代人类，而是在扩展文明的可能性边界。”"
                                "2025年的曙光中，AI智能体正将科幻照进现实，从云南梯田到深圳实验室，从手术台到直播间，这场技术革命正在重塑人类文明的底层逻辑。"
                                "当我们凝视智能体瞳孔中闪烁的代码之光，看到的不仅是冰冷的算法，更是人机协同进化的无限可能。人工智能作为21世纪最具颠覆性的技术力量，"
                                "正在从经济结构、社会形态到个体生活方式等各个层面深刻改变人类社会。这场智能革命既带来前所未有的机遇，也引发深层次的挑战，正在重新定义人类文明的发展轨迹。"
                                "经济结构的颠覆性重构正加速推进。制造业中智能机器人和自动化生产线使生产效率平均提升45%，产品不良率降低80%。某汽车厂商引入AI质检系统后，检测精度达到99.97%，"
                                "年节省质量成本超2亿元。新兴产业生态快速崛起，生成式AI催生的内容创作产业规模预计2025年将突破1000亿美元，AI训练师、数据标注师等新职业岗位年增长率达120%。"
                                "中国人工智能核心产业规模已超过5000亿元，带动相关产业规模超5万亿元。金融智能化转型成效显著，基于深度学习的信贷风控模型使银行不良贷款率下降35%，"
                                "智能投顾服务覆盖客户资产规模突破10万亿。部分券商应用AI客服系统后，服务响应时间从平均8分钟缩短至30秒内。社会运行机制的深度变革同样显著，城市治理智能化水平显著提升。"
                                "杭州“城市大脑”系统通过AI优化交通信号灯，使主干道通行效率提高15%，年均减少碳排放4.3万吨。北京部分社区试点AI网格员，事件处置效率提升60%。医疗健康革命挽救更多生命，"
                                "AI辅助诊断系统在肺结节识别等领域准确率达95%，超过资深医师水平。某三甲医院引入AI分诊后，急诊等待时间从52分钟降至18分钟，危急重症识别率提高40%。教育模式创新打破时空限制，"
                                "智能教育平台通过个性化学习路径规划，使学生平均成绩提升23%，学习时间减少30%。AI教师已能覆盖200多门学科的基础教学，服务偏远地区学生超500万人。"
                                "就业市场的结构性调整呈现双轨趋势，预计到2027年，全球将有8500万个工作岗位被AI替代，同时创造9700万个新岗位。数据分析师、AI训练师等新兴职业薪资水平已达传统岗位的2-3倍。"
                                        ), 
                            n_predict=100)
    
    def _cancel_thread():
        if test_generation_cancel:
            time.sleep(2)
            print("\nTest: task cancel request (thread)")
            send_request(triton_client, model_name, user_data, RequestType.TASK_CANCEL.value, 
                        request_id=DEFAULT_REQUEST_ID)
        else:
            time.sleep(0.1)
            print("\nTest: task abort request (thread)")
            send_request(triton_client, model_name, user_data, RequestType.TASK_ABORT.value, 
                        request_id=DEFAULT_REQUEST_ID)
    
    t1 = threading.Thread(target=_completions_thread)
    t2 = threading.Thread(target=_cancel_thread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


# ============================================================================
# 测试函数
# ============================================================================

def run_single_test(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    user_data: UserData,
    request_type: int,
    test_name: str,
    **kwargs
) -> bool:
    """运行单个测试用例"""
    try:
        success = send_request(client, model_name, user_data, request_type, **kwargs)
        if success:
            print(f"{test_name}: \033[92mSUCCESS\033[0m")
            return True
        else:
            print(f"{test_name}: \033[91mFAIL\033[0m")
            return False
    except Exception as e:
        print(f"{test_name}: \033[91mERROR - {e}\033[0m")
        return False


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Triton LlamaCpp Backend Client Test')
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output"
    )
    parser.add_argument(
        "-u", "--url",
        type=str,
        default="localhost:8001",
        help="Inference server URL (default: localhost:8001)"
    )
    parser.add_argument(
        "-m", "--model-name", type=str, default="llamacpp", help="Name of model"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    model_name = args.model_name
    user_data = UserData()
    
    try:
        with grpcclient.InferenceServerClient(
            url=args.url,
            verbose=args.verbose
        ) as triton_client:
            
            triton_client.load_model(model_name)
            if not triton_client.is_model_ready(model_name):
                print("FAILED : Load Model " + model_name + ", please input correct model name.")
                sys.exit(1)

            triton_client.start_stream(callback=partial(_callback, user_data))
            
            print("=" * 70)
            print("Starting LlamaCpp Backend Client Tests, " + model_name + " is testing...")
            print("=" * 70)
            
            # 测试计数器
            success_count = 0
            fail_count = 0
            total_tests = 0
            
            # ==================== 核心推理测试 ====================
            
            print("\n--- Core Inference Tests ---")
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.COMPLETIONS.value,
                "test completions request",
                temperature=0.8, top_k=50, n_predict=20,
                text_input="你好，请介绍一下你自己", stop="<|im_end|>"
            )
            success_count += success
            fail_count += 1 - success
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.OPENAI_COMPLETIONS.value,
                "test openai completions request",
                text_input=json.dumps(COMPLETION_REQUEST)
            )
            success_count += success
            fail_count += 1 - success
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.OPENAI_CHAT.value,
                "test openai chat request",
                text_input=json.dumps(CHAT_REQUEST)
            )
            success_count += success
            fail_count += 1 - success
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.TOKENIZE.value,
                "test tokenize request",
                request_id="request-1",
                text_input="你是谁"
            )
            success_count += success
            fail_count += 1 - success
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.DETOKENIZE.value,
                "test detokenize request",
                request_id="request-1",
                text_input="105043 100165"
            )
            success_count += success
            fail_count += 1 - success
            
            # ==================== 性能监控测试 ====================
            
            print("\n--- Performance Monitoring Tests ---")
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.METRICS.value,
                "test metrics request"
            )
            success_count += success
            fail_count += 1 - success
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.LORA_GET.value,
                "test lora get request"
            )
            success_count += success
            fail_count += 1 - success
            
            # ==================== Worker 状态管理测试 ====================
            
            print("\n--- Worker State Management Tests ---")
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.WORKER_SAVE.value,
                "test worker save request",
                request_id="0",
                text_input="worker-0"
            )
            success_count += success
            fail_count += 1 - success
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.WORKER_RESTORE.value,
                "test worker restore request",
                request_id="0",
                text_input="worker-0"
            )
            success_count += success
            fail_count += 1 - success
            
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.WORKER_ERASE.value,
                "test worker erase request",
                request_id="0"
            )
            success_count += success
            fail_count += 1 - success
            
            # ==================== 取消和终止测试 ====================
            
            print("\n--- Cancel and Abort Tests ---")
            total_tests += 2
            
            print("Test: cancel request (generation phase)")
            test_completions_with_cancel(triton_client, model_name, user_data, test_generation_cancel=True)
            success_count += 1
            
            print("Test: abort request (prefill phase)")
            test_completions_with_cancel(triton_client, model_name, user_data, test_generation_cancel=False)
            success_count += 1
            
            # ==================== 结果汇总 ====================
            
            print("\n" + "=" * 50)
            print(f"Test Summary: {success_count}/{total_tests} passed")
            print("=" * 50)
            
            if fail_count == 0:
                print("\033[92mALL TESTS PASSED\033[0m")
            else:
                print(f"\033[91m{fail_count} TESTS FAILED\033[0m")
            
            triton_client.stop_stream()
    
    except Exception as e:
        print(f"Encountered an error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 启动入口
# ============================================================================

if __name__ == "__main__":
    import threading  # 导入放在这里以避免循环导入
    
    main()
