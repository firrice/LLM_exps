import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from transformers import AutoTokenizer, Qwen2ForCausalLM
from threading import Thread
import torch

device = "cuda" # the device to load the model onto

class Solution(object):
    def __init__(self):
        self.model_path = "/media/data1/ubuntu_env/projects/LLM/download_LLMs/Qwen2-7B-Instruct-GPTQ-Int4"
        self.exp_data_txt_path = "/media/data1/ubuntu_env/projects/LLM/LLM_exps/NTK_RoPE_exp/NTK_exp_data.txt"
        self.split_token = "<pos>"
        self.gap_len = 1000

        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with open(self.exp_data_txt_path, 'r', encoding='utf-8') as f:
            self.exp_data = f.read()
    
    def prompt_init(self, chunk_data):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a summary for the information below:\n {}.".format(chunk_data)}
        ]
        self.text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate_stream(self, **generation_kwargs):
        self.model.generate(**generation_kwargs)

        
    def eval(self):
        exp_data_chunks = self.exp_data.split(self.split_token)
        prompt_data = ""
        for i in range(len(exp_data_chunks)):
            prompt_data += exp_data_chunks[i]
            if i == 9:
                self.prompt_init(prompt_data)
                model_inputs = self.tokenizer([self.text], return_tensors="pt").to(device)
                generation_kwargs = dict(model_inputs, streamer=self.streamer, max_new_tokens=512)

                thread = Thread(target=self.generate_stream, kwargs=generation_kwargs)
                thread.start()
                generated_text = ""
                position = 0
                # 流式输出
                print('LLM output:', end='', flush=True)
                for new_text in self.streamer:
                    torch.cuda.empty_cache()
                    generated_text += new_text
                    print(generated_text[position:], end='', flush=True)
                    position = len(generated_text)
                print('\n')
            

def main():
    solution = Solution()
    solution.eval()

if __name__ == "__main__":
    main()

