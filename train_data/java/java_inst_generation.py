import re
import openai
import time
import json
import codecs
import os

# 设定最大长度，超过此长度的数据将被跳过
MAX_CODE_LENGTH = 4096  # 4K字符，确保不会超出 GPT-3.5/4 的 token 限制

INST_TEMPLATE = """Please gain inspiration from the following buggy and its fix code snippets to create a high-quality programming problem. Present your output in two distinct sections: [Bug Description] and [Solution]. Your response should be concise and clear.

Buggy code snippet for inspiration:
```java
{buggy_code}
```

Fix code snippet for inspiration:
```java
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
    openai.api_key = ""
    openai.api_base = "" 
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
            
if __name__ == '__main__':
    #apr_original = codecs.open("./apr_original_data/apr_data_total.jsonl", "r", "utf-8")
    apr_original = codecs.open("../apr_original_data/after_java_322516.jsonl", "r", "utf-8")
    insts = []
    data_used = []
    if os.path.exists(f'apr_data_used.json'):
        print("Loading existing data used...")
        with open(f'apr_data_used.json', "r") as f:
            data_used = json.load(f)
    if os.path.exists(f'apr_instruction_total.json'):
        print("Loading existing instructions...")
        with open(f'apr_instruction_total.json', "r") as f:
            insts = json.load(f)

    i = 1 #计数
    for l in apr_original:
        if i % 100 == 0:
            print(f"The number of instructions is: {i}\n")

        if(i > 10000): break

        l = eval(l)
        if l in data_used:
            i+=1
            continue

        buggy_code = l["buggy_code"]
        fix_code = l["fix_code"]

        # **如果代码过长，跳过该条数据**
        if len(buggy_code) > MAX_CODE_LENGTH or len(fix_code) > MAX_CODE_LENGTH:
            # print(f"Skipping long input (Buggy: {len(buggy_code)} chars, Fix: {len(fix_code)} chars)")
            continue        
    
        gpt_input = INST_TEMPLATE.format(buggy_code=l["buggy_code"], fix_code=l["fix_code"])
        gpt_output = get_completion(gpt_input)

        matches = re.findall(INST_PATTERN, gpt_output, re.DOTALL | re.IGNORECASE)
        if matches:
            match = matches[0]
            bug_description = match[0] # not used
            bug_context = match[1]
            bug_reason = match[2]
            buggy_code_snippet = match[3]
            solution = match[4]# not used
            fix_code_snippet = match[5]
            fix_explanation = match[6]
            insts.append({
                'instruction': INST_HEAD.format(problem=bug_context + bug_reason + buggy_code_snippet),
                'input': "",
                'output': fix_code_snippet + fix_explanation
            })
            data_used.append(l)
        else:
            continue
        with open(f'apr_instruction_total.json', "w") as f:
            json.dump(insts, f, indent=2)
        with open(f'apr_data_used.json', "w") as f:
            json.dump(data_used, f, indent=2)

        i+=1
                    



        
