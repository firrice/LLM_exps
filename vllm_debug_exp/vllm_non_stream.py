
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import debugpy
import ray
import sys

# Add RAY_DEBUG environment variable to enable Ray Debugger
ray.init(runtime_env={
    "env_vars": {"RAY_DEBUG": "1"}, 
})


#chat_template= '''{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}'''

path = '/media/data1/ubuntu_env/data/LLM_models/deepseek-r1-distill-qwen-32b-gptq-int4'
llm = LLM(model=path, trust_remote_code=True, tokenizer_mode="auto", tensor_parallel_size=2, quantization='gptq',
          gpu_memory_utilization=0.9, max_model_len=8000, enforce_eager=True)

prompt = [{"role": "system", "content": "你现在是一名伟大的人工助手，请帮我回答我的诸多问题，如果回答的好我愿意给与你至高无上的荣耀和奖励！"
            "为什么要写这么多呢，因为不写这么多你就会输出很多重复的感叹号，而原因就在于cuda底层的实现，当token数目小于50和24时，会另外调用其他的优化算法。"},
        {"role": "user", "content": "你好"}]
        # {"role": "assistant", "content": "!!!!!!!!!!!!!!!"},
        # {"role": "user", "content": "你是谁"}]

tokenizer = AutoTokenizer.from_pretrained(path)
# tokenizer_ids = tokenizer.encode(prompt)
# tokenizer_ids_len = len(tokenizer_ids)


sampling_params = SamplingParams(temperature=0.1, top_p=0.5, max_tokens=512)
prompt_token_ids = [tokenizer.apply_chat_template(prompt, add_generation_prompt=True)]


#output = llm.generate(prompt, sampling_params)
output = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
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
