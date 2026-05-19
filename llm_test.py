#!/usr/bin/env python
'''
Copyright 2026. All Rights Reserved.
Author: Arthur Bin, JingliangGao
Date: 2026-05-11 15:38:20
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
import threading
import psutil
import os
import subprocess
from tritonclient.utils import InferenceServerException
from functools import wraps


try:
    import tritonclient.http as httpclient
    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    HTTP_CLIENT_AVAILABLE = False

# set global variables
timming_metrics = {}
local_metrics = {}
inference_start = 0
slient_used_host_memory = 0
slient_used_device_memory = 0
GLOBAL_METRICS = {
    "max_gpu_memory_mb": 0,
    "max_cpu_percent": 0,
    "max_used_memory_mb": 0,
}


# ============================================================================
# 硬件监测函数
# ============================================================================

def get_gpu_memory():
    """
    获取所有GPU已使用显存总和（MB）
    """
    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits"
        ]).decode().strip()

        values = [int(x) for x in result.split("\n") if x.strip()]

        return sum(values)

    except Exception:
        return 0


def get_used_memory():
    """
    获取 free -m 中 used 内存值（MB）
    """
    mem = psutil.virtual_memory()

    used_mb = (
        mem.total
        - mem.free
        - getattr(mem, "buffers", 0)
        - getattr(mem, "cached", 0)
    ) / 1024 / 1024

    return used_mb


def resource_monitor(interval=0.1):
    """
    interval:
        监控采样间隔（秒）
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            stop_event = threading.Event()

            max_gpu_memory = 0
            max_used_memory = 0
            max_cpu_percent = 0

            # 当前 Python 进程
            process = psutil.Process(os.getpid())

            # -------------------------------------------------
            # 后台监控线程
            # -------------------------------------------------

            def monitor():
            
                nonlocal max_cpu_percent
                nonlocal max_used_memory
                nonlocal max_gpu_memory

                for p in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if 'kytensor' in ' '.join(p.info['cmdline']):
                        # print("Triton PID : ", p.info['pid'])
                        triton_pid = p.info['pid']
                        process = psutil.Process(triton_pid)   # 监控 Triton 进程的资源占用

                process.cpu_percent(interval=None)

                while not stop_event.is_set():
                
                    time.sleep(interval)

                    # ------------------------
                    # CPU
                    # ------------------------

                    cpu_percent = process.cpu_percent(interval=None)

                    max_cpu_percent = max(
                        max_cpu_percent,
                        cpu_percent
                    )

                    # ------------------------
                    # Host Memory
                    # ------------------------

                    mem = psutil.virtual_memory()

                    used_memory = (
                        mem.total - mem.available
                    ) / 1024 / 1024

                    max_used_memory = max(
                        max_used_memory,
                        used_memory
                    )

                    # ------------------------
                    # GPU Memory
                    # ------------------------

                    gpu_memory = get_gpu_memory()

                    max_gpu_memory = max(
                        max_gpu_memory,
                        gpu_memory
                    )


            monitor_thread = threading.Thread(
                target=monitor
            )

            monitor_thread.daemon = True

            monitor_thread.start()

            # -------------------------------------------------
            # 执行目标函数
            # -------------------------------------------------

            try:
                result = func(*args, **kwargs)

            finally:

                stop_event.set()

                monitor_thread.join()

                # 写入全局变量
                GLOBAL_METRICS["max_gpu_memory_mb"] = round(
                    max_gpu_memory, 2
                )

                GLOBAL_METRICS["max_used_memory_mb"] = round(
                    max_used_memory, 2
                )

                GLOBAL_METRICS["max_cpu_percent"] = round(
                    max_cpu_percent, 2
                )

            return result

        return wrapper

    return decorator


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
    global timming_metrics

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
    token_index = 0

    
    while True:
        data_item = user_data._completed_requests.get()
        if isinstance(data_item, InferenceServerException):
            return False
        
        final_response = data_item.get_response().parameters.get('triton_final_response', {}).bool_param
        output_data = data_item.as_numpy("text_output")
         
        if output_data is not None and len(output_data) > 0:
            token_index += 1
            if token_index == 1 :
                first_token_time = time.time() - inference_start
                # print("[INFO] First token received, latency: {:.3f}s".format(first_token_time))
                local_metrics['first_token_time'] = first_token_time
                
            out_str = output_data[0][0].decode('utf-8')
            # print(out_str, end='', flush=True)
            answer += out_str

            # add performance data when 'timings' exists      JingliangGao 2026/05/13 
            out_dict = json.loads(out_str) 
            if 'timings' in list(out_dict.keys()):
                timming_metrics = out_dict

        
        output_token_data = data_item.as_numpy("token_output")
        if output_token_data is not None:
            metrics['tokens'] = output_token_data
        
        metrics_output = data_item.as_numpy("metrics_output")
        if metrics_output is not None:
            metrics['values'] = metrics_output
            # print("metrics:", metrics)  # 打印性能监控数据
        
        if final_response:
            break

    return True


def send_frequent_request(
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
            # print(out_str, end='', flush=True)
            answer += out_str
        
        output_token_data = data_item.as_numpy("token_output")
        if output_token_data is not None:
            metrics['tokens'] = output_token_data
        
        metrics_output = data_item.as_numpy("metrics_output")
        if metrics_output is not None:
            metrics['values'] = metrics_output
            # print("metrics:", metrics)  # 打印性能监控数据
        
        if final_response:
            break

    return True



# ============================================================================
# 测试函数
# ============================================================================
@resource_monitor(interval=0.01)
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
            # print(f"{test_name}: \033[92mSUCCESS\033[0m")
            return True
        else:
            # print(f"{test_name}: \033[91mFAIL\033[0m")
            return False
    except Exception as e:
        # print(f"{test_name}: \033[91mERROR - {e}\033[0m")
        return False


def run_throughput_test(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    user_data: UserData,
    request_type: int,
    test_name: str,
    num_requests: int = 10,
    **kwargs
) -> bool:
    """运行请求吞吐量测试，按顺序发送多个请求并计算平均请求率"""
    start_time = time.time()
    success_requests = 0

    for i in range(num_requests):
        success = send_frequent_request(client, model_name, user_data, request_type, **kwargs)
        if not success:
            break
        success_requests += 1

    elapsed = time.time() - start_time
    local_metrics['request_throughput_rps'] = round(
        success_requests / elapsed, 2
    ) if elapsed > 0 else 0
    local_metrics['request_throughput_total'] = num_requests
    local_metrics['request_throughput_success'] = success_requests
    local_metrics['request_throughput_time'] = round(elapsed, 3)
    local_metrics['request_throughput_test_name'] = test_name

    return success_requests == num_requests


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

def summary_data(metrics, local_metrics=local_metrics):
    
    # for key, value in metrics.items():
    #     print(f"{key}: {value}")
    
    # accquire keys with default value to avoid key error
    usage = metrics.get('usage', {})
    timings = metrics.get('timings', {})
    
    # 性能维度
    model_load_time = local_metrics.get('model_load_time', 0)
    model_unload_time = local_metrics.get('model_unload_time', 0)
    first_token_time = local_metrics.get('first_token_time', 0)
    inference_cost_time = local_metrics.get('inference_cost_time', 0)
    prompt_tokens = usage.get('prompt_tokens', 0)
    prompt_per_second = timings.get('prompt_per_second', 0)
    prefill_time = prompt_tokens / prompt_per_second if prompt_per_second > 0 else 0
    decode_tokens = usage.get('completion_tokens', 0)
    decode_per_second = timings.get('predicted_per_second', 0)
    decode_time = decode_tokens / decode_per_second if decode_per_second > 0 else 0
    request_throughput = local_metrics.get('request_throughput_rps', 0)
    throughput_total = local_metrics.get('request_throughput_total', 0)
    throughput_success = local_metrics.get('request_throughput_success', 0)
    throughput_time = local_metrics.get('request_throughput_time', 0)

    # 内存维度
    host_mem_peak = local_metrics.get('host_mem_peak_mb', 0)
    host_mem_load = local_metrics.get('host_mem_cost_mb', 0)
    device_mem_peak = local_metrics.get('device_mem_peak_mb', 0)
    device_mem_cost = local_metrics.get('device_mem_cost_mb', 0)

    # CPU维度
    cpu_peak = local_metrics.get('cpu_peak_percent', 0)

    print("|" + "-"*100 + "|")
    print(f"  [性能维度]  | 模型加载耗时(s) : {model_load_time:.3f}, 模型卸载耗时(s) : {model_unload_time:.3f} "                                             )
    print(f"              | 首次Token延迟耗时(s) : {first_token_time:.3f}, 单次完整推理耗时(s) : {inference_cost_time:.3f}" )
    print(f"              | Prefill阶段耗时(s) : {prefill_time:.3f}, Token生成速度(tokens/s) : {prompt_per_second:.3f}"   )
    print(f"              | Decode阶段耗时(s) : {decode_time:.3f},  Token生成速度(tokens/s) : {decode_per_second:.3f}"    )
    print(f"              | 请求吞吐率(req/s) : {request_throughput:.2f} (" \
          f"{throughput_success}/{throughput_total} requests in {throughput_time:.3f}s)")
    print(f"  [内存维度]  | 模型加载内存占用(Mb) :  {host_mem_load:.3f}"                                                        )
    print(f"              | 推理峰值内存占用(Mb) : {host_mem_peak:.3f}"                                                     )
    print(f"              | 显存占用 : {device_mem_cost:.3f} , 显存峰值 : {device_mem_peak:.3f}"                                )
    print(f"  [CPU维度]   | CPU占用率 : {cpu_peak:.2f}% "                                                                     )
    print("|" + "-"*100 + "|")

def main():
    """主函数"""
    args = parse_args()
    model_name = args.model_name
    user_data = UserData()
    global inference_start
    global local_metrics
    global slient_used_host_memory
    global slient_used_device_memory
    
    try:
        with grpcclient.InferenceServerClient(
            url=args.url,
            verbose=args.verbose
        ) as triton_client:

            # 测试前的系统资源监测
            mem = psutil.virtual_memory()
            slient_used_host_memory = (
                mem.total
                - mem.free
                - getattr(mem, "buffers", 0)
                - getattr(mem, "cached", 0)
            ) / 1024 / 1024

            
            try:
                gpu_cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i", "0"
                ]
                slient_used_device_memory = int(subprocess.check_output(gpu_cmd).decode().strip())
            except subprocess.CalledProcessError:
                slient_used_device_memory = 0

            # 加载模型
            model_load_start = time.time()
            triton_client.load_model(model_name)
            model_load_time = time.time() - model_load_start

            mem = psutil.virtual_memory()
            model_used_host_memory = (
                mem.total
                - mem.free
                - getattr(mem, "buffers", 0)
                - getattr(mem, "cached", 0)
            ) / 1024 / 1024

            try:
                gpu_cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i", "0"
                ]
                model_used_device_memory = int(subprocess.check_output(gpu_cmd).decode().strip())
            except subprocess.CalledProcessError:
                model_used_device_memory = 0

            local_metrics['model_load_time']    = model_load_time
            local_metrics['host_mem_cost_mb']   = model_used_host_memory   - slient_used_host_memory
            local_metrics['device_mem_cost_mb'] = model_used_device_memory - slient_used_device_memory


            if not triton_client.is_model_ready(model_name):
                print("[ERROR] FAILED : Load Model " + model_name + ", please input correct model name.")
                sys.exit(1)

            triton_client.start_stream(callback=partial(_callback, user_data))
            
            
            # 测试计数器
            success_count = 0
            fail_count = 0
            total_tests = 0
            
            # ==================== 核心推理测试 ====================
            
            print("[INFO] Start to test inference ... ")
            
            inference_start = time.time()
            print("[INFO] Sending one inference request ... ")
            total_tests += 1
            success = run_single_test(
                triton_client, model_name, user_data,
                RequestType.OPENAI_COMPLETIONS.value,
                "test openai completions request",
                text_input=json.dumps(COMPLETION_REQUEST)
            )
            inference_cost_time = time.time() - inference_start
            print(f"[INFO] Single inference completed, total cost time: {inference_cost_time:.3f}s")     
            local_metrics['inference_cost_time'] = inference_cost_time      
            
            success_count += success
            fail_count += 1 - success
            
            
            # ==================== 请求吞吐量测试 ====================
            total_tests += 1
            multiple_inference_start = time.time()
            print("[INFO] Sending multiple inference requests ... ")
            success = run_throughput_test(
                triton_client, model_name, user_data,
                RequestType.OPENAI_COMPLETIONS.value,
                "test request throughput",
                num_requests=10,
                text_input=json.dumps(COMPLETION_REQUEST)
            )
            multiple_inference_cost_time = time.time() - multiple_inference_start
            print(f"[INFO] Multiple inference completed, total cost time: {multiple_inference_cost_time:.3f}s")
            success_count += success
            fail_count += 1 - success

            # ==================== 性能监控测试 ====================

            # total_tests += 1
            # success = run_single_test(
            #     triton_client, model_name, user_data,
            #     RequestType.METRICS.value,
            #     "test metrics request"
            # )
            # success_count += success
            # fail_count += 1 - success

            # unload model after test
            model_unload_start = time.time()
            triton_client.unload_model(model_name)
            model_unload_time = time.time() - model_unload_start
            local_metrics['model_unload_time'] = model_unload_time


            # ===================== 测试结果总结 ====================
            if GLOBAL_METRICS.get('max_used_memory_mb') is not None:
                local_metrics['cpu_peak_percent']   = GLOBAL_METRICS['max_cpu_percent']
                local_metrics['host_mem_peak_mb']   = GLOBAL_METRICS['max_used_memory_mb'] - slient_used_host_memory
                local_metrics['device_mem_peak_mb'] = GLOBAL_METRICS['max_gpu_memory_mb']  - slient_used_device_memory

            print("[INFO] Summary data ... ")
            summary_data(timming_metrics, local_metrics)
            
            triton_client.stop_stream()
    
    except Exception as e:
        print(f"[ERROR] Encountered an error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================cle
# 启动入口
# ============================================================================

if __name__ == "__main__":
    import threading  # 导入放在这里以避免循环导入
    
    main()
