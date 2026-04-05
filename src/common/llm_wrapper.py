import os
import openai
from openai import Stream, ChatCompletion
import re

GPT3 = "gpt-3.5-turbo-16k-0613"
GPT4 = "gpt-4-turbo"
GPT4O_MINI = "gpt-4o-mini"
GPT4O = "gpt-4o"
GPT4_1 = "gpt-4.1"
LLAMA3 = "llama3.2:latest"
LLAMA3_3_70B = "llama-3.3-70b-instruct"
RWKV = "rwkv"
QWEN = "qwen2.5:7b-instruct-fp16"
QWEN_2_5_32B = "qwen2.5-32b-instruct"
QWEN_2_5_72B = "qwen2.5-72b-instruct"
INTERN = "internlm/internlm2.5:7b-chat-1m"
INTERN_2_5 = "internlm2.5-latest"
INTERN_3 = "internlm3-latest"
GEMMA2 = "gemma2:9b-instruct-fp16"
DEEPSEEKR1_8B = "deepseek-r1:8b-llama-distill-fp16"
DEEPSEEKR1_32B = "deepseek-r1:32b-qwen-distill-fp16"
DEEPSEEKR1_32B_DASHSCOPE = "deepseek-r1-distill-qwen-32b"
DEEPSEEKR1 = "deepseek-ai/DeepSeek-R1"
QWQ_32B_LOCAL = "qwq:32b-fp16"
QWQ_32B = "Qwen/QwQ-32B"
QWEN_OMNI_7B = "qwen2.5-omni-7b"
GLM_4_FLASH_0414 = "glm-4-flash-250414"
QWEN_2_5_72B_SILICONFLOW = "Qwen/Qwen2.5-72B-Instruct"
QWEN_2_5_32B_SILICONFLOW = "Qwen/Qwen2.5-32B-Instruct"
QWEN_2_5_7B_SILICONFLOW = "Qwen/Qwen2.5-7B-Instruct" # Free
QWEN_2_5_7B_SILICONFLOW_PRO = "Pro/Qwen/Qwen2.5-7B-Instruct"
QWEN_3_32B = "Qwen/Qwen3-32B"
QWEN_3_14B = "Qwen/Qwen3-14B"
QWEN_3_8B = "Qwen/Qwen3-8B"
GLM_4_32B_0414 = "THUDM/GLM-4-32B-0414"
GLM_4_9B_0414 = "THUDM/GLM-4-9B-0414"
GLM_5 = "Pro/zai-org/GLM-5"
GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"
QWEN_3_5_122A10_SILICONFLOW = "Qwen/Qwen3.5-122B-A10B"
QWEN_3_5_397A17_SILICONFLOW = "Qwen/Qwen3.5-397B-A17B"
QWEN_3_5_122A10 = "qwen3.5-122b-a10b"
QWEN_3_5_397A17 = "qwen3.5-397b-a17b"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default="token-abc123")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", default="token-abc123")
INTERN_API_KEY = os.getenv("INTERN_API_KEY", default="token-abc123")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", default="token-abc123")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", default="token-abc123")
GENSTUDIO_API_KEY = os.getenv("GENSTUDIO_API_KEY", default="token-abc123")

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.vllm_client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        self.gpt_client = openai.OpenAI(
            base_url='https://api.openai-proxy.org/v1',
            api_key=OPENAI_API_KEY,
        )
        self.dashscope_client = openai.OpenAI(
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key=DASHSCOPE_API_KEY,
        )
        self.intern_client = openai.OpenAI(
            base_url='https://chat.intern-ai.org.cn/api/v1/',
            api_key=INTERN_API_KEY,
        )
        self.zhipu_client = openai.OpenAI(
            base_url='https://open.bigmodel.cn/api/paas/v4/',
            api_key=ZHIPU_API_KEY,
        )
        self.rwkv_client = openai.OpenAI(
            base_url='http://localhost:8000',
            api_key="token-abc123",
        )
        self.siliconflow_client = openai.OpenAI(
            base_url='https://api.siliconflow.cn/v1',
            api_key=SILICONFLOW_API_KEY,
        )
        self.genstudio_client = openai.OpenAI(
            base_url='https://cloud.infini-ai.com/maas/v1',
            api_key=GENSTUDIO_API_KEY,
        )
        
        self.vllm_client_memory = openai.OpenAI(
            base_url="http://10.171.189.127:8000/v1",
            api_key="token-abc123",
        )
        self.vllm_client_judger = openai.OpenAI(
            base_url="http://10.171.189.127:8001/v1",
            api_key="token-abc123",
        )
    
    def get_client(self, model_name):
        if model_name == RWKV:
            return self.rwkv_client
        elif model_name == GPT4 or model_name == GPT4O_MINI or model_name == GPT3 or model_name == GPT4O or model_name == GPT4_1 or model_name == GEMINI_2_5_FLASH_LITE:
            return self.gpt_client
        elif model_name == QWEN_2_5_72B or model_name == DEEPSEEKR1_32B_DASHSCOPE or model_name == QWEN_2_5_32B or model_name == QWEN_OMNI_7B or model_name == QWEN_3_5_397A17 or model_name == QWEN_3_5_122A10:
            return self.dashscope_client
        elif model_name == INTERN_2_5 or model_name == INTERN_3:
            return self.intern_client
        elif model_name == GLM_4_FLASH_0414:
            return self.zhipu_client
        elif model_name == QWEN_2_5_72B_SILICONFLOW or model_name == QWEN_2_5_32B_SILICONFLOW or model_name == QWEN_2_5_7B_SILICONFLOW or model_name == QWEN_2_5_7B_SILICONFLOW_PRO \
            or model_name == DEEPSEEKR1 or model_name == QWEN_3_32B or model_name == QWEN_3_14B or model_name == QWEN_3_8B or model_name == GLM_4_32B_0414 or model_name == GLM_4_9B_0414 \
            or model_name == QWQ_32B or model_name == GLM_5 or model_name == QWEN_3_5_122A10_SILICONFLOW or model_name == QWEN_3_5_397A17_SILICONFLOW:
            return self.siliconflow_client
        elif model_name == LLAMA3_3_70B:
            return self.genstudio_client
        elif model_name == "Qwen2_5_VL_Memory":
            return self.vllm_client_memory
            # return self.vllm_client
        elif model_name == "Qwen2_5_VL_Judger":
            return self.vllm_client_judger
        else:
            return self.vllm_client
    def request_with_token_count(self, prompt, system_prompt=None, model_name=LLAMA3, stream=False):
        client = self.get_client(model_name)
        if client == self.dashscope_client:
            stream = False if model_name == QWEN_3_5_397A17 or model_name == QWEN_3_5_122A10 or model_name == QWEN_2_5_72B else True

        if stream:
            if system_prompt is not None:
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },{
                        "role": "user", 
                        "content": prompt
                    }
                ]
            else:
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            if model_name == QWEN_3_8B or model_name == QWEN_3_14B or model_name == QWEN_3_32B or model_name == GLM_5 or model_name == QWEN_3_5_397A17 or model_name == QWEN_3_5_122A10 or model_name == QWEN_3_5_397A17_SILICONFLOW or model_name == QWEN_3_5_122A10_SILICONFLOW:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    stream=stream,
                    extra_body={"enable_thinking": False}
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    stream=stream
                )
            content = ""
            for chunk in response:
                # print(chunk)
                if chunk.choices:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                else:
                    usage = chunk.usage
                    tokens = (usage.prompt_tokens, usage.total_tokens, usage.completion_tokens)
        else:
            if system_prompt is not None:
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },{
                        "role": "user", 
                        "content": prompt
                    }
                ]
            else:
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            if model_name == QWEN_3_8B or model_name == QWEN_3_14B or model_name == QWEN_3_32B or model_name == GLM_5 or model_name == QWEN_3_5_397A17 or model_name == QWEN_3_5_122A10 or model_name == QWEN_3_5_397A17_SILICONFLOW or model_name == QWEN_3_5_122A10_SILICONFLOW:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    stream=stream,
                    extra_body={"enable_thinking": False}
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    stream=stream
                )
            content = response.choices[0].message.content
            # Intern will return None for usage
            if response.usage is not None:
                tokens = (response.usage.prompt_tokens, response.usage.total_tokens, response.usage.completion_tokens)
            else:
                tokens = (0, 0, 0)
        return content, tokens