import json
import os
import re
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from tqdm import tqdm
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<|system|>", "</|system|>"

BENCHMARK_PROMPT = """
The problem is as follows.Its bug_type is {bug_type}.It should achieve the function :{docstring}\n
Please write the fixed code of the following buggy code, and the fixed code must be between ``` and ``` :\n

```java
{problem}\n
```
"""

class LoRA_CodeLlamaChat:
    def __init__(self, model_id: str, lora_weights_path: str = None, device: str = "cuda"):
        """初始化 CodeLlama 并加载 LoRA 权重"""
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        if lora_weights_path:
            self.load_lora_weights(lora_weights_path)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def load_lora_weights(self, lora_weights_path: str):
        """使用 PEFT 加载 LoRA 微调权重"""
        print(f"🔄 正在加载 LoRA 权重: {lora_weights_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_weights_path, torch_dtype=torch.float16,)
        self.model = self.model.to(self.device)
        print("✅ LoRA 权重加载成功！")
    
    def chat_completion_and_generation(
        self,
        dialogs: List[List[dict]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = 256,
        num_responses: int = 1,  # 指定生成回答的数量
    ):
        """Generate assistant responses for multiple conversational dialogs."""
        if max_gen_len is None:
            max_gen_len = 256  # 默认限制生成长度
        
        prompt_tokens = []
        
        for dialog in dialogs:
            # 处理 system 指令（如果有的话）
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
                    }
                ] + dialog[2:]

            # 处理对话编码
            dialog_tokens = []
            for i in range(0, len(dialog) - 1, 2):
                prompt, answer = dialog[i], dialog[i + 1]
                encoded_text = self.tokenizer.encode(
                    f"{B_INST} {prompt['content'].strip()} {E_INST} {answer['content'].strip()} ",
                    add_special_tokens=True,
                )
                dialog_tokens.extend(encoded_text)
            
            # 处理最后一条用户消息
            assert dialog[-1]["role"] == "user", f"Last message must be from user, got {dialog[-1]['role']}"
            last_user_message = self.tokenizer.encode(
                f"{B_INST} {dialog[-1]['content'].strip()} {E_INST}",
                add_special_tokens=True,
            )
            dialog_tokens.extend(last_user_message)
            prompt_tokens.append(dialog_tokens)
        #至此，prompt_tokens生成完毕，分词完成

        # 转换为 PyTorch Tensor，并移动到 GPU
        input_tensors = [torch.tensor(t).unsqueeze(0).to(self.device) for t in prompt_tokens]

        # 生成对话回复
        results = []
        for input_ids in input_tensors:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=(input_ids != self.tokenizer.pad_token_id).long(),  # 显式传递 mask
                    max_length=min(self.model.config.max_position_embeddings, len(input_ids[0]) + max_gen_len),
                    do_sample=True,  # 开启随机采样
                    temperature=temperature,  # 设定温度参数
                    top_p=top_p,  # 设定 Top-P 采样
                    num_return_sequences=num_responses,  # 保持单个返回序列
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            input_length = input_ids.shape[1]
            generated_texts = []
            
            for gen_seq in generated_ids:
                response_tokens = gen_seq[input_length:]  # 只取生成部分
                response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                generated_texts.append(response_text)

            results.append({"generation": [{"role": "assistant", "content": text} for text in generated_texts]})

        return results

def main():
    # 设定 JSON 文件路径
    json_file_path = "/data/capito/a_bishe/bench/java_humanevalpack.jsonl" #bench
    output_file_path = "/data/capito/a_bishe/LoRA_infer/java_LoRA_fixed_tasks.json"  # 纯二维列表格式
    used_json_path = "/data/capito/a_bishe/LoRA_infer/java_LoRA_used_tasks.json"  # 记录已处理的 task_id

    # 初始化模型
    model_id = "/data/share/code-llama/CodeLlama-7b-Instruct-hf"
    lora_weights_path = "/data/capito/a_bishe/train/output4/checkpoint-32103"
    chat_model = LoRA_CodeLlamaChat(model_id, lora_weights_path)

    # for name, param in chat_model.model.named_parameters():
    #     if torch.isnan(param).any() or torch.isinf(param).any():
    #         print(f"❌ 发现 NaN/Inf 参数: {name}")

    # chat_model.model.to("cpu")  # 把模型转移到 CPU
    # chat_model.device = "cpu"

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
                dialogs = [
                    [
                        {"role": "system", "content": "Write a solution to the following coding problem"},
                        {"role": "user", "content": prompt},
                    ]
                ]
                
                # 运行前记录 GPU 占用
                torch.cuda.reset_peak_memory_stats()
                
                responses = chat_model.chat_completion_and_generation(dialogs, num_responses=10) ######################
                
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                gpu_usages.append(gpu_memory)

                extracted_responses = []
                for response_set in responses:
                    for response in response_set["generation"]:
                        response_text = response["content"]
                        extracted_code = re.findall(r"```java\n(.*?)\n```", response_text, re.DOTALL)

                        if not extracted_code:
                            extracted_code = re.findall(r"'''java\n(.*?)\n'''", response_text, re.DOTALL)

                        if not extracted_code:
                            extracted_code = re.findall(r"```\n(.*?)\n```", response_text, re.DOTALL)

                        if not extracted_code:
                            extracted_code = re.findall(r"'''\n(.*?)\n'''", response_text, re.DOTALL)

                        if not extracted_code:
                            if "java\n" in response_text:
                                extracted_code = [response_text.split("java\n", 1)[1].strip()]

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

if __name__ == "__main__":
    main()
