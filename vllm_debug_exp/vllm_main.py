
import os
import importlib
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import asyncio
import time

class Solution(object):
    def __init__(self):
        self.model_file_dict = {"deepseek_R1_Distill_Qwen_32B_GPTQ_int4_g128": "/media/data1/ubuntu_env/data/LLM_models/deepseek-r1-distill-qwen-32b-gptq-int4"}
        self.model_name = "deepseek_R1_Distill_Qwen_32B_GPTQ_int4_g128"
        self.query = "你是谁"
        # ==============model init=================
        models_module =  importlib.import_module("models")
        self.model_solution = getattr(models_module, self.model_name)(model_root=self.model_file_dict[self.model_name])
        # ==============query preprocess, which was attached to the specific model========
        self.model_solution.query_preprocess(self.query)
    
    async def forward(self):
        try:
            self.model_solution.current_generation_task = asyncio.create_task(self.model_solution.generate_streaming())
            await self.model_solution.current_generation_task  # end util the current generation
            await self.model_solution._cleanup_engine_resources() # clean up all resources related to vllm engines and ray
        except asyncio.TimeoutError:
            print("\n[System] Generation timeout")
        except asyncio.CancelledError:
            print("\n[System] Process interrupted by user")
        finally:
            pass
        
async def main():
    solution = Solution()
    await solution.forward()

if __name__ == "__main__":
    asyncio.run(main())