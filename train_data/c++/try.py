import re
import openai
import time
import json
import codecs
import os
from concurrent.futures import ThreadPoolExecutor

# 设定最大长度，超过此长度的数据将被跳过
MAX_CODE_LENGTH = 4096  # 4K字符，确保不会超出 GPT-3.5/4 的 token 限制
NUM_THREADS = 5  # 线程数（等于数据总量 / 每个线程处理的数据量）

INST_TEMPLATE = """Please gain inspiration from the following buggy and its fix code snippets to create a high-quality programming problem. Present your output in two distinct sections: [Bug Description] and [Solution]. Your response should be concise and clear.

Buggy code snippet for inspiration:
```c
{buggy_code}
```

Fix code snippet for inspiration:
```c
{fix_code}
```

Guidelines for each section:

1. [Bug Description]: This should be **completely self-contained**.  There are three subsections in this section:[Bug context], [Bug reason], and [Buggy code snippet]. First provide all the contextual information one needs to understand the buggy code snippet. Second, provide the reason of the bug in the buggy code  snippet. At last,  output the buggy code snippet. Note: Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.

2. [Solution]: There are twosubsections in this section:[Fix code snippet] and [Fix explanation]. 
First output the fix code snippet, then according to the fix code snippet, offer a comprehensive, **correct** explanation how to accurately addresses the [Bug Description] provided.

"""

INST_PATTERN = r'\[Bug Description\](.*?)\[Bug context\](.*?)\[Bug reason\](.*?)\[Buggy code snippet\](.*?)\[Solution\](.*?)\[Fix code snippet\](.*?)\[Fix explanation\](.*)'

INST_HEAD = """Write a solution to the following coding problem:
{problem}"""
def get_completion(prompt, model="gpt-3.5-turbo", retries=1, timeout=5, delay=2):
    openai.api_key = "9dd20911b7644f868a610b12d9ce461a"
    openai.api_base = "https://lgcchat.openai.azure.com/" 
    openai.api_type = "azure"
    openai.api_version = "2024-08-01-preview"  
    deployment_name = "chat"  
    messages = [{"role": "system", "content": "You are an expert coder."}, {"role": "user", "content": prompt}]
    for attempt in range(retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=1.0,
                engine=deployment_name, # add engine
            )
            return response.choices[0].message["content"]
        except Exception as e:
            if attempt < retries:
                print(f"Timeout occurred, retrying after {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                print(f"Failed to get completion: {e}")
                raise e
            
import threading

file_lock = threading.Lock()

def process_data(data_chunk, thread_id): 
    """每个线程单独写入一个 JSON 文件"""
    insts = [] 
    data_used = []

    for i, l in enumerate(data_chunk):
        l = eval(l)
        
        buggy_code = l["buggy_code"]
        fix_code = l["fix_code"]

        if len(buggy_code) > MAX_CODE_LENGTH or len(fix_code) > MAX_CODE_LENGTH:
            print(f"[Thread-{thread_id}] Skipping long input")
            continue        

        gpt_input = INST_TEMPLATE.format(buggy_code=buggy_code, fix_code=fix_code)
        gpt_output = get_completion(gpt_input)

        matches = re.findall(INST_PATTERN, gpt_output, re.DOTALL | re.IGNORECASE)
        if matches:
            match = matches[0]
            bug_context = match[1]
            bug_reason = match[2]
            buggy_code_snippet = match[3]
            fix_code_snippet = match[5]
            fix_explanation = match[6]
            insts.append({
                'instruction': INST_HEAD.format(problem=bug_context + bug_reason + buggy_code_snippet),
                'input': "",
                'output': fix_code_snippet + fix_explanation
            })
            data_used.append(l)

    # 每个线程写入单独文件
    thread_inst_file = f'apr_instruction_total_{thread_id}.json'
    thread_data_file = f'apr_data_used_{thread_id}.json'

    with open(thread_inst_file, "w") as f:
        json.dump(insts, f, indent=2)

    with open(thread_data_file, "w") as f:
        json.dump(data_used, f, indent=2)

    print(f"[Thread-{thread_id}] Finished processing {len(data_chunk)} data points.")

if __name__ == '__main__':
    apr_original = codecs.open("../apr_original_data/after_c_138499.jsonl", "r", "utf-8")
    data_lines = apr_original.readlines()[:10000]
    chunk_size = 2000
    data_chunks = [data_lines[i:i + chunk_size] for i in range(0, len(data_lines), chunk_size)]

    num_threads = min(NUM_THREADS, len(data_chunks))

    print(f"Starting {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_id, data_chunk in enumerate(data_chunks):
            futures.append(executor.submit(process_data, data_chunk, thread_id))

        for future in futures:
            future.result()

    print("All threads have completed execution.")

    # 🔒 合并所有 JSON 文件
    final_insts = []
    final_data_used = []

    for thread_id in range(num_threads):
        thread_inst_file = f'apr_instruction_total_{thread_id}.json'
        thread_data_file = f'apr_data_used_{thread_id}.json'

        if os.path.exists(thread_inst_file):
            with open(thread_inst_file, "r") as f:
                final_insts.extend(json.load(f))
            os.remove(thread_inst_file)

        if os.path.exists(thread_data_file):
            with open(thread_data_file, "r") as f:
                final_data_used.extend(json.load(f))
            os.remove(thread_data_file)

    # 🔒 统一写入主文件
    with open("apr_instruction_total.json", "w") as f:
        json.dump(final_insts, f, indent=2)

    with open("apr_data_used.json", "w") as f:
        json.dump(final_data_used, f, indent=2)

    print("All JSON files have been merged.")
                        



        