
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径
model_path = "/media/data1/ubuntu_env/data/LLM_models/deepseek-r1-distill-qwen-32b-gptq-int4"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 使用FP16精度
    trust_remote_code=True,
    device_map="auto"  # 自动分配到可用设备
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 输入文本
prompts = [
    "9.11和9.8哪个数字大",
    "如果你是人，你最想做什么",
    "How many e in word deepseek",
    "There are ten birds in a tree. A hunter shoots one. How many are left in the tree?"
]

# 构造输入
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

# 推理生成
outputs = model.generate(
    input_ids=inputs["input_ids"].to(model.device),
    attention_mask=inputs["attention_mask"].to(model.device),
    max_length=512,  # 最大生成长度
    num_return_sequences=1,
    do_sample=False
)

# 解码输出
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# 打印结果
for prompt, generated_text in zip(prompts, generated_texts):
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)