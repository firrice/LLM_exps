
import json
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_path = "/media/data1/ubuntu_env/projects/LLM/download_LLMs/Qwen2-7B-Instruct"
jl_file_path = '/media/data1/ubuntu_env/projects/LLM/LLM_exps/exp_data/CASSum-main/dataset.jl'
old_context_len = 32768
alpha = 8
save_txt_path = "/media/data1/ubuntu_env/projects/LLM/LLM_exps/NTK_RoPE_exp/NTK_exp_data.txt"



tokenizer = AutoTokenizer.from_pretrained(model_path)
new_context_len = old_context_len * alpha

token_count = 0
token_gap = 1000
old_txt_content = ""
save_txt = open(save_txt_path, 'w')
# 打开.jl文件
with open(jl_file_path, 'r', encoding='utf-8') as file:
    # 逐行读取文件
    for line in file:
        # 移除行尾的换行符并解析JSON
        json_object = json.loads(line.strip())
        json_content = json_object["text"]
        content_token_num = len(tokenizer.encode(json_content))
        token_count += content_token_num
        old_txt_content += json_content
        if token_count >= new_context_len:
            print(token_count)
            break
        
    # post-process: 每隔1K个token添加一个标识符<pos>
    new_txt_content = ""
    tmp_content = tokenizer.encode(old_txt_content)
    for i in range(new_context_len // token_gap):
        chunk_content = tmp_content[i * token_gap : (i + 1) * token_gap]
        chunk_content = "".join(tokenizer.decode(chunk_content)) + "<pos>"
        new_txt_content += chunk_content
    new_txt_content.strip("<pos>")
    save_txt.write(new_txt_content)
    save_txt.close()
