
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import debugpy
#debugpy.listen(("127.0.0.1", 5678))
#debugpy.wait_for_client()


# prompt = '''
# {"role": "system", "content": "You are a helpful assistant."},
# {"role": "user", "content": "please answer the question below: \n"
# '''
# prompt += "你是谁？}"

prompt = '''
    <指令>请作为大模型助手回答用户的问题。</指令>\n
    history: \n
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "!!!!!!!!!!!!!!!"}。
    <问题> 你好 </问题>\n
'''

sampling_params = SamplingParams(temperature=0.1, top_p=0.5, max_tokens=512)
path = '/media/data1/ubuntu_env/projects/LLM/download_LLMs/Qwen2-7B-Instruct-GPTQ-Int4'
llm = LLM(model=path, trust_remote_code=True, tokenizer_mode="auto", tensor_parallel_size=1, dtype="auto")

output = llm.generate(prompt, sampling_params)
generated_text = output[0].outputs[0].text
print(f"Generated text: {generated_text!r}")

# clear_command = 'clear'
# stop_query = "stop"
# welcome_prompt = "欢迎使用 ChatGLM3-6B-chat 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
# while True:
#     query = input("\n用户：")
#     if query.strip() == stop_query:
#         break
#     if query.strip() == "clear":
#         os.system(clear_command)
#         print(welcome_prompt)
#         continue
#     output = llm.generate(query, sampling_params)
#     generated_text = output[0].outputs[0].text
#     print(f"Generated text: {generated_text!r}")
