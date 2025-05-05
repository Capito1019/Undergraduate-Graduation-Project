import torch
import json
import os
import re
import time
import moe_peft
from typing import List, Optional
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BENCHMARK_PROMPT = """
[INST]
<|system|>Write a solution to the following coding problem.</|system|>
The problem is as follows.Its bug_type is {bug_type}.It should achieve the function :{docstring}\n
Please write the fixed code of the following buggy code, and the fixed code must be between ``` and ``` :\n

```cpp
{problem}\n
```
[/INST]
"""

def model_and_tokenizer(
    base_model_path,
    adapter_path=None,
    device=None,
    load_16bit=True,
    load_8bit=False,
    load_4bit=False,
    flash_attn=False,
):
        # 加载模型
    model = moe_peft.LLMModel.from_pretrained(
        base_model_path,
        device=device or moe_peft.executor.default_device_name(),
        attn_impl="flash_attn" if flash_attn else "eager",
        bits=(8 if load_8bit else (4 if load_4bit else None)),
        load_dtype=torch.bfloat16 if load_16bit else torch.float32,
    )

    tokenizer = moe_peft.Tokenizer(base_model_path)

    # 加载 adapter
    if adapter_path:
        model.load_adapter(adapter_path, "default")
    else:
        model.init_adapter(moe_peft.AdapterConfig(adapter_name="default"))
    
    return model, tokenizer

import collections

def run_inference_on_samples(
    model,
    tokenizer,
    prompt_template=None,
    samples=None,
    max_new_tokens=256,
    temperature=0.6,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1,
    repeat_times=1,
):
    generation_config = moe_peft.GenerateConfig(
        adapter_name="default",
        prompt_template=prompt_template,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    results = []
    expert_stats_percent_list = []
    expert_stats_total_counter = collections.Counter()

    for instruction, input_text in samples:
        generation_config.prompts = [(instruction.strip(), input_text.strip() or None)]

        for _ in range(repeat_times):
            generate_params = {
                "model": model,
                "tokenizer": tokenizer,
                "configs": [generation_config],
                "max_gen_len": max_new_tokens,
            }

            output, experts_stats = moe_peft.generate(**generate_params)
            results.append(output['default'][0])

            stats = experts_stats.get('default', {})
            total = sum(stats.values())
            if total == 0:
                percent_stats = {eid: 0.0 for eid in stats}
            else:
                percent_stats = {eid: (count / total) * 100 for eid, count in stats.items()}

            expert_stats_percent_list.append(percent_stats)
            expert_stats_total_counter.update(stats)

            # print(f"🔍 Expert usage for current prompt (%):")
            # for eid in sorted(percent_stats):
            #     print(f"  Expert {eid}: {percent_stats[eid]:.2f}%")

    return results, expert_stats_percent_list, expert_stats_total_counter

# 示例调用
def main():
    # 设定 JSON 文件路径
    json_file_path = "/data/capito/a_bishe/bench/cpp_humanevalpack.jsonl" #bench
    output_file_path = "/data/capito/a_bishe/Moe_LoRA_infer/cpp_MoeLoRA_fixed_tasks.json"  # 纯二维列表格式
    used_json_path = "/data/capito/a_bishe/Moe_LoRA_infer/cpp_MoeLoRA_used_tasks.json"  # 记录已处理的 task_id

    # 初始化模型
    base_model_path = "/data/share/code-llama/CodeLlama-7b-Instruct-hf"
    adapter_path = "/data/capito/a_bishe/Moe_LoRA_train/MoE-PEFT/MoE-PEFT/APR_mola_1/APR_mola_1_12844"
    model, tokenizer = model_and_tokenizer(base_model_path = base_model_path, adapter_path = adapter_path)

    all_prompt_expert_percent = []
    overall_expert_counter = collections.Counter()
    

        # 读取已完成任务
    if os.path.exists(used_json_path):
        with open(used_json_path, "r", encoding="utf-8") as f:
            used_tasks = set(json.load(f))
    else:
        used_tasks = set()

    # 读取已修复的 solutions
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as f:
            fixed_solutions = json.load(f)
    else:
        fixed_solutions = []

    # 记录已处理任务数量（用于避免重复）
    processed_count = len(fixed_solutions)

    # 记录总运行时间
    start_time = time.time()
    num_processed = 0
    gpu_usages = []  # 记录 GPU 使用量

    # 读取 JSONL 文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total_tasks = len(lines)

    # 使用 tqdm 进度条
    with tqdm(total=total_tasks, desc="Processing Tasks", unit="task") as pbar:
        for idx, line in enumerate(lines):
            try:
                data = json.loads(line)
                task_id = data.get("task_id", "").strip()
                if not task_id:
                    print("Skipping entry with missing task_id.")
                    pbar.update(1)
                    continue

                # 跳过已处理的任务
                if task_id in used_tasks or idx < processed_count:
                    print(f"Skipping already processed task: {task_id}")
                    pbar.update(1)
                    continue

                bug_type = data.get("bug_type", "").strip()
                docstring = data.get("docstring", "").strip()
                declaration = data.get("declaration", "").strip()
                buggy_solution = data.get("buggy_solution", "").strip()

                if not declaration or not buggy_solution:
                    print("Skipping invalid entry.")
                    pbar.update(1)
                    continue

                problem = f"{declaration}\n\n{buggy_solution}"
                prompt = BENCHMARK_PROMPT.format(bug_type=bug_type, docstring=docstring, problem=problem)

               # 生成修复代码
                dialogs = [(prompt, "")]

                # 运行前记录 GPU 占用
                torch.cuda.reset_peak_memory_stats()
                
                responses, percent_stats_list, expert_counter = run_inference_on_samples(
                    model = model,
                    tokenizer = tokenizer,
                    prompt_template="APR",
                    samples=dialogs,
                    repeat_times=10,
                )
                all_prompt_expert_percent.extend(percent_stats_list)
                overall_expert_counter.update(expert_counter)

                gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                gpu_usages.append(gpu_memory)

                extracted_responses = []
                for response_text in responses:
                    extracted_code = re.findall(r"```cpp\n(.*?)\n```", response_text, re.DOTALL)

                    if not extracted_code:
                        extracted_code = re.findall(r"'''cpp\n(.*?)\n'''", response_text, re.DOTALL)

                    if not extracted_code:
                        extracted_code = re.findall(r"```\n(.*?)\n```", response_text, re.DOTALL)

                    if not extracted_code:
                        extracted_code = re.findall(r"'''\n(.*?)\n'''", response_text, re.DOTALL)

                    if not extracted_code:
                        if "cpp\n" in response_text:
                            extracted_code = [response_text.split("cpp\n", 1)[1].strip()]

                    if not extracted_code:
                        if "'''\n" in response_text:
                            extracted_code = [response_text.split("'''\n", 1)[1].strip()]
                    
                    if not extracted_code:
                        if "```\n" in response_text:
                            extracted_code = [response_text.split("```\n", 1)[1].strip()]
                        else:
                            extracted_code = [response_text.strip()]  # 作为最后的备选方案，保留原始内容
                    
                    if extracted_code:
                        extracted_responses.append(extracted_code[0])  # 修复点：确保非空内容被记录             

                if extracted_responses:
                    # 追加到 fixed_solutions（仅存修复代
            
                    fixed_solutions.append(extracted_responses)

                    # 更新已处理任务
                    used_tasks.add(task_id)

                    # 立即写入 JSON，防止中途崩溃丢失数据
                    with open(output_file_path, "w", encoding="utf-8") as output_file:
                        json.dump(fixed_solutions, output_file, indent=4, ensure_ascii=False)

                    with open(used_json_path, "w", encoding="utf-8") as f:
                        json.dump(list(used_tasks), f, indent=4, ensure_ascii=False)

                num_processed += 1
                pbar.update(1)

            except json.JSONDecodeError:
                print("Skipping invalid JSON line.")
                pbar.update(1)

    # 计算 GPU 统计
    max_gpu_usage = max(gpu_usages) if gpu_usages else 0
    avg_gpu_usage = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0

    # 打印统计信息
    total_time = time.time() - start_time
    print(f"Fixed solutions saved to: {output_file_path}")
    print(f"Total tasks processed: {num_processed} / {total_tasks}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Max GPU Memory Usage: {max_gpu_usage:.2f} MB")
    print(f"Avg GPU Memory Usage: {avg_gpu_usage:.2f} MB")

    # 汇总专家使用比例
    total_count = sum(overall_expert_counter.values())
    print("\n📊 Final Expert Usage Percentages (All Prompts):")
    if total_count == 0:
        print("No expert usage data recorded.")
    else:
        for eid in sorted(overall_expert_counter):
            percent = (overall_expert_counter[eid] / total_count) * 100
            print(f"  Expert {eid}: {percent:.2f}%")

if __name__ == "__main__":
    main()

                



