
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import asyncio
import ray
import time
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

class deepseek_R1_Distill_Qwen_32B_GPTQ_int4_g128(object):
    def __init__(self, model_root):
        self.engine_args = AsyncEngineArgs(model=model_root, 
                                    enforce_eager=True, tensor_parallel_size=2, gpu_memory_utilization=0.9, quantization='gptq',
                                    max_model_len=8000)
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.5, max_tokens=512)
        self.current_generation_task = None
    
    def query_preprocess(self, query):
        self.prompt = f'''
            <指令>请作为大模型助手回答用户的问题。</指令>\n
            history: \n
            [{{"role": "user", "content": "你好"}},
            {{"role": "assistant", "content": "!!!!!!!!!!!!!!!"}}]。
            <问题> {query} </问题>\n
        '''
        #self.prompt = query

    async def generate_streaming(self):
        try:
            results_generator = self.model.generate(self.prompt, self.sampling_params, request_id=time.monotonic())
            previous_text = ""
            async for request_output in results_generator:
                text = request_output.outputs[0].text
                print(text[len(previous_text):], end='', flush=True)
                previous_text = text
        except asyncio.exceptions.CancelledError:
            print("streaming progress has been interrupted by user.")
        except Exception as e:
            print(f"error in knowledge chat: {e}")
        finally:
            await self._cleanup_generation_resources()
    
    # clean up resources related to the current generation
    async def _cleanup_generation_resources(self):
        if self.current_generation_task and not self.current_generation_task.done():
            self.current_generation_task.cancel()
            try:
                await self.current_generation_task   # cancel() is also anther kind of op, so need to be awaited too!
            except asyncio.CancelledError:
                pass
        print("\n[System] Resources related to current generation released")
        self.current_generation_task = None
    
    # clean up resources related to VLLM engine and Ray
    async def _cleanup_engine_resources(self):
        if self.model._background_loop_unshielded is not None:
            self.model._background_loop_unshielded.cancel()
            self.model._background_loop_unshielded = None
        self.model.background_loop = None
        ray.shutdown()


async def main():
    model_solution = deepseek_R1_Distill_Qwen_32B_GPTQ_int4_g128("/media/data1/ubuntu_env/data/LLM_models/deepseek-r1-distill-qwen-32b-gptq-int4")
    model_solution.query_preprocess("你是谁")
    
    try:
        model_solution.current_generation_task = asyncio.create_task(model_solution.generate_streaming())
        await model_solution.current_generation_task  # end util the current generation
        await model_solution._cleanup_engine_resources() # clean up all resources related to vllm engines and ray
    except asyncio.TimeoutError:
        print("\n[System] Generation timeout")
    except asyncio.CancelledError:
        print("\n[System] Process interrupted by user")
    finally:
        pass

if __name__ == "__main__":
    asyncio.run(main())
    