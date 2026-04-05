import os
import numpy as np
import openai
from openai import Stream, ChatCompletion
import re

LLAMA3V = "llama3.2-vision:11b-instruct-q8_0"
MINICPM = "minicpm-v:8b-2.6-fp16"
GPT4O_V = "gpt-4o"
GPT4O_MINI_V = "gpt-4o-mini"
GPT4_1 = "gpt-4.1"
GPT5_MINI = "gpt-5-mini"
INTERN_VL = "OpenGVLab/InternVL2_5-8B"
INTERN_VL_2_5 = "internvl2.5-78b"
INTERN_VL_3 = "internvl3-78b"
QWEN_VL_7B = "qwen2.5-vl-7b-instruct"
QWEN_VL_32B = "qwen2.5-vl-32b-instruct"
QWEN_VL_72B = "qwen2.5-vl-72b-instruct"
QWEN_OMNI_7B = "qwen2.5-omni-7b"
GLM_4V_FLASH = "glm-4v-flash"
QWEN_VL_3B_LOCAL = "Qwen_3B"
QWEN_VL_3B_O3DVQA = "Qwen_3B_O3DVQA"
QWEN_VL_7B_LOCAL = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_VL_7B_O3DVQA = "Qwen_O3DVQA"
QWEN_VL_72B_SILICONFLOW = "Qwen/Qwen2.5-VL-72B-Instruct"
QWEN_VL_32B_SILICONFLOW = "Qwen/Qwen2.5-VL-32B-Instruct"
QWEN_VL_7B_SILICONFLOW = "Pro/Qwen/Qwen2.5-VL-7B-Instruct"
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

import base64
from io import BytesIO
from PIL import Image
import cv2

def image_convert_color(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(im_rgb)
    return image

def image_to_base64(image, save_path=None):
    # image = image_convert_color(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    if save_path is not None:
        image.save(save_path, format="PNG")
    img_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
    return img_str

class VLMWrapper:
    def __init__(self, temperature=0.0):
        self.image_id = 0
        self.ollama_client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
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
        self.vllm_client = openai.OpenAI(
            base_url='http://localhost:8000/v1',
            api_key="token-abc123",
        )
        self.siliconflow_client = openai.OpenAI(
            base_url='https://api.siliconflow.cn/v1',
            api_key=SILICONFLOW_API_KEY,
        )
        self.lmdeploy_client = openai.OpenAI(
            base_url='http://localhost:23333/v1',
            api_key="token-abc123",
        )
    
    def get_client(self, model_name):
        if model_name == GPT4O_V or model_name == GPT4O_MINI_V or model_name == GPT4_1 or model_name == GPT5_MINI:
            return self.gpt_client
        elif model_name == LLAMA3V or model_name == MINICPM:
            return self.ollama_client
        elif model_name == QWEN_VL_7B or model_name == QWEN_VL_72B or model_name == QWEN_VL_32B or model_name == QWEN_OMNI_7B or model_name == QWEN_3_5_122A10 or model_name == QWEN_3_5_397A17:
            return self.dashscope_client
        elif model_name == INTERN_VL_2_5 or model_name == INTERN_VL_3:
            return self.intern_client
        elif model_name == GLM_4V_FLASH:
            return self.zhipu_client
        elif model_name == QWEN_VL_32B_SILICONFLOW or model_name == QWEN_VL_72B_SILICONFLOW or model_name == QWEN_VL_7B_SILICONFLOW or model_name == QWEN_3_5_122A10_SILICONFLOW or model_name == QWEN_3_5_397A17_SILICONFLOW:
            return self.siliconflow_client
        else: # model_name == QWEN_VL_7B_LOCAL or model_name == QWEN_VL_7B_O3DVQA or model_name == QWEN_VL_3B_LOCAL or model_name == QWEN_VL_3B_O3DVQA:
            return self.vllm_client
        
    def request_with_token_count(self, prompt, model_name=LLAMA3V, image=None, stream=False, multi_sentence=False, save_path=None):
        client = self.get_client(model_name)

        if model_name == QWEN_OMNI_7B:
            stream = True

        if image is not None:
            self.image_id += 1
            if isinstance(image, Image.Image):
                image = image_to_base64(image, save_path=save_path)
            else:
                image = image_to_base64(cv2.imread(image), save_path=save_path)

        if stream:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image
                                }
                            }
                        ]
                    }
                ],
                stream=stream,
                stream_options={
                    "include_usage": True
                }
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
            if client == self.vllm_client:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image
                                    }
                                }
                            ]
                        }
                    ],
                    stream=stream, 
                    timeout=60000,
                    extra_body={'repetition_penalty': 1.2}
                )
            else:
                if model_name == QWEN_3_5_122A10_SILICONFLOW or model_name == QWEN_3_5_397A17_SILICONFLOW or model_name == QWEN_3_5_122A10 or model_name == QWEN_3_5_397A17:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image
                                        }
                                    }
                                ]
                            }
                        ],
                        stream=stream, 
                        extra_body={"enable_thinking": False}
                    )
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image
                                        }
                                    }
                                ]
                            }
                        ],
                        stream=stream, 
                    )
            content = response.choices[0].message.content
            if response.usage is not None:
                tokens = (response.usage.prompt_tokens, response.usage.total_tokens, response.usage.completion_tokens)
            else:
                tokens = (0, 0, 0)
        return content, tokens