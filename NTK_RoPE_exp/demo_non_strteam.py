import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from modelscope import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto


class Solution(object):
    def __init__(self):
        self.model_path = "/media/data1/ubuntu_env/projects/LLM/download_LLMs/Qwen2-7B-Instruct"
        self.exp_data_txt_path = "/media/data1/ubuntu_env/projects/LLM/LLM_exps/NTK_RoPE_exp/NTK_exp_data.txt"
        self.split_token = "<pos>"
        self.gap_len = 1000

        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
                )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
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
        
    def eval(self):
        exp_data_chunks = self.exp_data.split(self.split_token)
        prompt_data = ""
        for i in range(len(exp_data_chunks)):
            prompt_data += exp_data_chunks[i]
            if i != 1: continue
            self.prompt_init(prompt_data)
            model_inputs = self.tokenizer([self.text], return_tensors="pt").to(device)
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(response)
            a = 1

            

def main():
    solution = Solution()
    solution.eval()

if __name__ == "__main__":
    main()

