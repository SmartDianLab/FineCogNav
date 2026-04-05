from dataclasses import dataclass, field, asdict
import re
from typing import Any, Dict, Optional, Union, List, DefaultDict
import os
import sys
from pathlib import Path

import cv2
sys.path.append(str(Path(str(os.getcwd())).resolve()))
import time
import math
import random
import json
import json_repair
import numpy as np
from pathlib import Path

from utils.logger import logger
from utils.vision import Frame, VisionClient

from src.common.param import args
from src.common.llm_wrapper import LLMWrapper
from src.common.vlm_wrapper import VLMWrapper


actions_description = """TASK_FINISH
MOVE_FORWARD (5 meters)
TURN_LEFT (15 degrees)
TURN_RIGHT (15 degrees)
ASCEND (2 meters)
DESCEND (2 meters)
MOVE_LEFT (5 meters)
MOVE_RIGHT (5 meters)"""


class CognitiveAgentConfig:
    class ConfigBase:
        def __init__(self):
            self.model = ""
    class PerceptionConfig(ConfigBase):
        def __init__(self):
            super().__init__()
            self.detector = "vlm"
            self.vlm_model = ""
    class AttentionConfig(ConfigBase):
        def __init__(self):
            super().__init__()
            self.use_memory = True
    class MemoryConfig(ConfigBase):
        def __init__(self):
            super().__init__()
    class ImageryConfig(ConfigBase):
        def __init__(self):
            super().__init__()
    class ProblemSolvingConfig(ConfigBase):
        def __init__(self):
            super().__init__()
    class ReasoningConfig(ConfigBase):
        def __init__(self):
            super().__init__()
    class DecisionMakingConfig(ConfigBase):
        def __init__(self):
            super().__init__()
    def __init__(self):
        self.perception = self.PerceptionConfig()
        self.attention = self.AttentionConfig()
        self.memory = self.MemoryConfig()
        self.imagery = self.ImageryConfig()
        self.problem_solving = self.ProblemSolvingConfig()
        self.reasoning = self.ReasoningConfig()
        self.decision_making = self.DecisionMakingConfig()

@dataclass
class EnvState:
    instruction_text: str = ""
    sub_instructions: List[Dict[str, Any]] = None
    episode_id: str = ""
    trajectory_id: str = ""
    scenes_id: int = 0
    macro_step: int = 0
    collision_warning: str = ""
    attention_questions: List[List[Dict[str, Any]]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    current_observation: Dict[str, Any] = field(default_factory=dict)
    current_instruction: Dict[str, Any] = field(default_factory=dict)
    next_instruction: Dict[str, Any] = field(default_factory=dict)
    instruction_index: int = 0
    subgoals: List[List[str]] = field(default_factory=list)
    current_subgoal: Optional[str] = None
    next_subgoal: Optional[str] = None
    subgoal_index: int = 0
    reference_imagination: List[List[str]] = field(default_factory=list)
    subgoal_memory: List[List[Any]] = field(default_factory=list)
    step_memory: List[List[List[Any]]] = field(default_factory=list)
    log_dir_base: str = None
    log_dir: str = None

    def get_valid_actions(self) -> str:
        valid_actions= """0: TASK_FINISH
1: MOVE_FORWARD (5 meters)
2: TURN_LEFT (15 degrees)
3: TURN_RIGHT (15 degrees)
4: ASCEND (2 meters)
5: DESCEND (2 meters)
6: MOVE_LEFT (5 meters)
7: MOVE_RIGHT (5 meters)"""
        if self.instruction_index == len(self.sub_instructions) - 1:
            return valid_actions
        else:
            return valid_actions.replace("0: TASK_FINISH", "").strip()

    def add_step_memory(self, memory: str):
        while len(self.step_memory) <= self.instruction_index:
            self.step_memory.append([])
        while len(self.step_memory[self.instruction_index]) <= self.subgoal_index:
            self.step_memory[self.instruction_index].append([])
        self.step_memory[self.instruction_index][self.subgoal_index].append((self.macro_step, memory))
    
    def add_subgoal_memory(self, memory: str):
        while len(self.subgoal_memory) <= self.instruction_index:
            self.subgoal_memory.append([])
        self.subgoal_memory[self.instruction_index].append((self.current_subgoal, memory))

    def get_current_subgoal_memory(self) -> str:
        memory_list = []
        if len(self.step_memory) > self.instruction_index and len(self.step_memory[self.instruction_index]) > self.subgoal_index:
            for _, (step, memory) in enumerate(self.step_memory[self.instruction_index][self.subgoal_index]):
                memory_list.append(f"At step-{step}: {memory}")
        memory_str = "\n".join(memory_list)
        memory_str = memory_str.strip()
        if len(memory_str) == 0:
             memory_str = "No memory yet."
        return memory_str
    
    def get_instruction_memory(self) -> str:
        memory_list = []
        if len(self.subgoal_memory) > self.instruction_index:
            for _, (subgoal, memory) in enumerate(self.subgoal_memory[self.instruction_index]):
                memory_list.append(f"For {subgoal}: {memory}")
        memory_str = "\n".join(memory_list)
        memory_str += f"\nFor {self.current_subgoal}: "
        memory_str += self.get_current_subgoal_memory()
        return memory_str.strip()

    def get_state_dict(self) -> Dict[str, Any]:
        dict = asdict(self)
        return dict

context = EnvState()

# suggestion
class Attention:
    def __init__(self, global_config: CognitiveAgentConfig):
        self.global_config = global_config
        config = global_config.attention
        self.llm = LLMWrapper()
        self.model_name = config.model
    ### Attention Module: Vision Suggestion 
    def perception_attention(self, log_dir=None):
        start_time = time.time()
        if len(context.attention_questions) > context.instruction_index:
            return context.attention_questions[context.instruction_index]
        current_instruction = context.current_instruction
        current_instruction_text = current_instruction['sub-instruction']
        current_landmark = current_instruction['landmark'].copy()
        current_instruction = current_instruction_text + "\nLandmark: " + " | ".join(current_landmark)
        next_instruction = context.next_instruction
        if next_instruction is not None:
            next_instruction_text = next_instruction['sub-instruction'] if next_instruction is not None else 'None'
            next_landmark = next_instruction['landmark'].copy()
            next_instruction = next_instruction_text + "\nLandmark: " + " | ".join(next_landmark)
        else:
            next_instruction_text = 'None'
        prompt = """[COGNITIVE ROLE]: Attention

[TASK]: You are an attention guider for a drone's vision module. Generate questions considering distinct landmarks in [Instructions] to direct visual attention of Perception Module.

[INPUT]: 
<Current Instruction>
<Next Instruction>

[OUTPUT CONSTRAINTS]: 
1. Strictly formatted as a JSON code block without any explanatory text.
2. The OUTPUT JSON must include the following information:
A list (array) where each element is an object with the following contents:
  - "landmark": The name of the landmark.
  - "question": A question regarding the landmark's CURRENT status(e.g. position or distance, etc.), guiding the attention of the perception module.

[NOTE]:
1. Question examples:
  - "Is the <landmark> in front of you / on your left / on your right / above / below?"
  - "Is the <landmark> far / near?"
  - "What are the distinct attributes(color, shape, size, etc.) of the <landmark>?"
2. The questions should be specific to the CURRENT status of the landmark and should not reference any actions or future states.

[EXPECTED OUTPUT]:
```json
[
  {{
  "landmark": "landmark 1",
  "question": "...?"
  }},
  ...
]
```

### Input ###

<Current Instruction>:
{current_instruction}

<Next Instruction>:
{next_instruction}"""
        prompt = prompt.format(current_instruction=current_instruction, next_instruction=next_instruction)
        response_raw, tokens = self.llm.request_with_token_count(prompt, model_name=self.model_name)
        response = re.findall(r"```json(?:\w+)?\n(.*?)```", response_raw, re.DOTALL | re.IGNORECASE)
        if len(response) == 0:
            try: 
                suggestion = json_repair.loads(response_raw)
            except Exception as e:
                suggestion = response_raw
        else:
            suggestion = json_repair.loads(response[-1])
        if log_dir is not None:
            end_time = time.time()
            file_name = "suggestion.txt"
            with open(os.path.join(log_dir, file_name), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(response_raw)
                f.write("\n---\n")
                f.write(str(suggestion))
                f.write("\n---\n")
                prompt_tokens, total_tokens, completion_tokens = tokens
                f.write(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, time: {int((end_time - start_time)*1000)}")
        context.attention_questions.append(suggestion)
        return suggestion

# perception
class Perception:
    def __init__(self, global_config: CognitiveAgentConfig, attention:Attention = None):
        self.global_config = global_config
        config = global_config.perception
        self.llm = LLMWrapper()
        self.detector = config.detector
        self.model_name = config.model
        self.vlm_model = config.vlm_model
        self.attention = attention
        self.vision = VisionClient(self.detector, vlm_model=self.vlm_model)
    def get_scene(self, rgb, reget=False, log_dir=None, img_path=None):
        prompt = """[COGNITIVE ROLE]: Perception

[TASK]: You are an advanced multimodal perception system for a drone that navigates in the real world. Your task is to analyze first-person view(the drone's camera is facing forward) image and generate environmental semantics.

[INPUT]:
<Attention>: The landmarks you should focus on

[NOTE]:
1. ONLY include CURRENT VISIBLE landmarks in the <Attention>.

[OUTPUT CONSTRAINTS]:
1. Strictly formatted as a JSON code block without any explanatory text.
2. The [OUTPUT] JSON must include the following information:
- "overall": A string describe the scene according to the image input.
- "details": A string describe the VISIBLE landmarks in the <Attention>, estimate their the location and distance(in meters). In Front: describe the middle part of the image, On my Left: describe the left part of the image, On my Right: describe the right part of the image, etc.

[EXPECTED OUTPUT]:
```json
{{
  "overall": "Overall: I see a scene of ...",
  "details": "In Front: ... On my Left: ... On my Right: ... etc."
}}
```

### Input ###

<Attention>:
{suggestion}"""
        start_time = time.time()
        _suggestions = self.attention.perception_attention(log_dir=log_dir)
        suggestion = []
        for _suggestion in _suggestions:
            suggestion.append(f"{_suggestion['question']}")
        suggestion = "\n".join(suggestion)
        prompt = prompt.format(suggestion=suggestion)
        observation_raw, tokens = self.vision.detect_capture_with_token_count(rgb=rgb, prompt=prompt, save_path=img_path)
        observations = re.findall(r"```json(?:\w+)?\n(.*?)```", observation_raw, re.DOTALL | re.IGNORECASE)
        if len(observations) == 0:
            observation = observation_raw
            try:
                observation = json_repair.loads(observation)
                overall = observation['overall']
                details = observation['details']
                observation = overall + "\n" + details
            except Exception as e:
                observation = observation_raw
        else: 
            observation = json_repair.loads(observations[-1])
            overall = observation['overall']
            details = observation['details']
            if type(details) == dict:
                _details = ""
                for k, v in details.items():
                    _details += f"{k}: {v}\n"
                details = _details
            elif type(details) == list:
                _details = ""
                for d in details:
                    _details += f"{d}\n"
                details = _details
            observation = overall + "\n" + details
        if log_dir is not None:
            end_time = time.time()
            file_name = "scene.txt" if not reget else "scene_reget.txt"
            with open(os.path.join(log_dir, file_name), 'w+') as f:
                f.write(self.vlm_model)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(observation_raw)
                f.write("\n---\n")
                f.write(str(observation))
                f.write("\n---\n")
                prompt_tokens, total_tokens, completion_tokens = tokens
                f.write(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, time: {int((end_time - start_time)*1000)}")
        context.current_observation = observation
        return observation
    
# history manager
class Memory:
    def __init__(self, global_config: CognitiveAgentConfig, attention:Attention = None):
        self.global_config = global_config
        config = global_config.memory
        self.llm = LLMWrapper()
        self.model_name = config.model
        self.history_actions = []
        self.history_observations = []
        self.history = None
        self.attention = attention
        self.step_memory = None
        self.subgoal_memory = {}
        self.instruction_memory = {}
    def _get_memory(self, memory: dict, subgoal: str):
        if subgoal in memory.keys():
            return memory[subgoal]
        return None
    def _append_step_memory(self, subgoal: str, value: str, step:int):
        if subgoal in self.subgoal_memory.keys():
            self.subgoal_memory[subgoal].append((step, value))
        else:
            self.subgoal_memory[subgoal] = [(step, value)]
    def get_subgoal_memory(self, subgoal=None):
        memory = self._get_memory(self.subgoal_memory, subgoal)
        if memory is None:
            memory = []
        return memory
    def get_subgoal_memory(self, subgoal=None, length=0, log_dir=None):
        memory = self._get_memory(self.subgoal_memory, subgoal)
        if memory is None:
            memory = []
        if length == -1:
            length = len(memory)
        memory = memory[-length:] if length > 0 else memory
        subgoal_memory = ""
        for i, (step, item) in enumerate(memory):
            if i < length:
                subgoal_memory += f"At step-{step}: {item}\n"
        if len(subgoal_memory) == 0:
            subgoal_memory = "None"
        if log_dir is not None:
            file_name = "subgoal_memory.txt"
            with open(os.path.join(log_dir, file_name), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(f"{subgoal_memory.strip()}")
        return subgoal_memory.strip()
    def get_instruction_memory(self, subgoals=None, log_dir=None):
        memory = ""
        if subgoals is None:
            return None
        else:
            for i, subgoal in enumerate(subgoals):
                memory_item = self._get_memory(self.instruction_memory, subgoal)
                if memory_item is not None:
                    memory += f"For <{subgoal}>: {memory_item}"
            if log_dir is not None:
                file_name = "instruction_memory_without_current_subgoal.txt"
                with open(os.path.join(log_dir, file_name), 'w+') as f:
                    f.write(self.model_name)
                    f.write("\n---\n")
                    f.write(f"{memory.strip()}")
            return memory.strip()

    def generate_step_memory(self, log_dir=None):
        current_instruction = context.current_instruction
        next_instruction = context.next_instruction
        current_landmarks = current_instruction['landmark'].copy()
        next_landmarks = next_instruction['landmark'].copy() if next_instruction is not None else []
        landmarks = current_landmarks.extend(next_landmarks)
        observation = context.current_observation
        subgoal = context.current_subgoal
        start_time = time.time()
        prompt = """[COGNITIVE ROLE]: Memory

[TASK]: You are a **Memory Manager** for a drone that navigates in the real world. Generate a memory statement using the provided [INPUT] data.

[INPUT]:
<Current Observation>: The scene you are *SEEING*
<Current Action>: The action you are *DOING*
<Landmarks>

[NOTE]:
1. Keep accurate DEGREE/DISTANCE in <Action>
2. ONLY includes VISIBLE <Landmarks> in the <Current Observation>
3. Do NOT infer or include any planning steps or actions not explicitly mentioned in the input.

[OUTPUT CONSTRAINTS]:
1. Strictly formatted as a JSON code block without any explanatory text
2. **Word limit**: < 30 words

[EXPECTED OUTPUT]:
```json
{{
  "step_memory": the memory in the form of "I see ...(observation); I ...(action)".
}}
```

### Input ###

<Current Observation>:
{observation}

<Current Action>:
{action}

<Landmarks>:
{landmarks}"""

        responses_raw = ''
        try: 
            history = self.get_tuple(1)
            if len(history) == 0:
                observation = None
                subgoal = None
            else:
                observation = history[0]['observation']
                action = history[0]['action']
                history = {
                    "observation": observation,
                    "action": action
                }
            prompt = prompt.format(observation=history['observation'], action=history['action'], landmarks=landmarks)
            responses_raw, tokens = self.llm.request_with_token_count(prompt, model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            if len(responses) == 0:
                response = json_repair.loads(responses_raw)
            else:
                response = json_repair.loads(responses[-1])
            self.working_memory_raw = responses_raw
            self.step_memory = response['step_memory']
            context.add_step_memory(self.step_memory)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to parse response: {responses_raw}")
            tokens = (0, 0, 0)
        
        if log_dir is not None:
            end_time = time.time()
            with open(os.path.join(log_dir, 'step_memory.txt'), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(responses_raw)
                f.write("\n---\n")
                f.write(json.dumps(self.step_memory))
                f.write("\n---\n")
                prompt_tokens, total_tokens, completion_tokens = tokens
                f.write(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, time: {int((end_time - start_time)*1000)}")
    def update_instruction_memory(self, log_dir=None):
        current_instruction = context.current_instruction
        next_instruction = context.next_instruction
        landmarks = current_instruction['landmark'].copy()
        next_landmarks = next_instruction['landmark'].copy() if next_instruction is not None else []
        landmarks.extend(next_landmarks)
        start_time = time.time()
        prompt = """[COGNITIVE ROLE]: Memory

[TASK]: You are a **Memory Manager** for a drone that navigates in the real world. **Consolidate** <Raw Subgoal Memory> containing a list of actions and observations in temporal order.

[INPUT]:
<Raw Subgoal Memory>
<Landmarks>

[NOTE]:
1. **Merge the Information:**
  - Merge the same actions(note that turn left and turn right are different actions) and accumulate the numerical values.
  - Only merge *consecutive* actions of the same type (for example, consecutive "move forward" steps).
    - Do not merge actions if they are interrupted by a different action type.
    - Preserve the temporal order of actions and observations.
  - Only include the landmarks that are VISIBLE in the <Raw Subgoal Memory>, and pay special attention to changes in landmarks..
2. **Output Requirements:**
  - The output memory should be concise, clear, and logically organized.

[OUTPUT CONSTRAINTS]:
1. Strictly formatted as a JSON code block without any explanatory text
2. **Word limit**: < 40 words

[EXPECTED OUTPUT]:
```json
{{
  "subgoal_memory": "I have seen ...; I ...(action)".
}}
```

### Input ###
<Raw Subgoal Memory>:
{raw_memory}

<Landmarks>:
{landmarks}"""
        responses_raw = ''
        try: 
            subgoal_memory = context.get_current_subgoal_memory()
            prompt = prompt.format(raw_memory=subgoal_memory, landmarks=landmarks)
            responses_raw, tokens = self.llm.request_with_token_count(prompt, model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            if len(responses) == 0:
                response = json_repair.loads(responses_raw)
            else:
                response = json_repair.loads(responses[-1])
            subgoal_memory_compressed = response['subgoal_memory']
            context.add_subgoal_memory(subgoal_memory_compressed)
        except Exception as e:
            import traceback
            traceback.print_exc()
            subgoal_memory_compressed = responses_raw
            context.add_subgoal_memory(subgoal_memory_compressed)
            logger.error(f"Failed to parse response: {responses_raw}")
        if log_dir is not None:
            end_time = time.time()
            with open(os.path.join(log_dir, 'subgoal_memory_compress.txt'), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(responses_raw)
                f.write("\n---\n")
                f.write(subgoal_memory_compressed)
                f.write("\n---\n")
                prompt_tokens, total_tokens, completion_tokens = tokens
                f.write(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, time: {int((end_time - start_time)*1000)}")
    
    def add_tuple(self, observation, action):
        actions = actions_description.split('\n')
        self.history_observations.append(observation)
        self.history_actions.append(actions[action])
    
    def get_tuple(self, length=3):
        history = []
        observations = self.history_observations[-length:]
        actions = self.history_actions[-length:]
        for observation, action in zip(observations, actions):
            item = {}
            item['observation'] = observation
            item['action'] = action
            history.append(item)
        return history

    def clear(self):
        self.history_actions = []
        self.history_observations = []
        self.history = None
        self.step_memory = None
        self.subgoal_memory = {}
        self.instruction_memory = {}

# collision estimation
class Imagery:
    def __init__(self, global_config: CognitiveAgentConfig, attention:Attention = None):
        self.global_config = global_config
        self.config = global_config.imagery
        self.llm = LLMWrapper()
        self.model_name = self.config.model
        self.attention = attention
    def _check_collision(self, depth_img, action, img_width=672, img_height=672, drone_width=1.0, drone_height=0.1, fov=90, distance=5.1):
        # print(depth_img.shape) # (360, 640, 1)
        pixel_angle = fov / img_width
        center_x = img_width // 2
        center_y = img_height // 2
        if action == 1:
            half_angle_x = np.arctan(drone_width / (2 * distance)) * (180 / np.pi)
            half_angle_y = np.arctan(drone_height / (2 * distance)) * (180 / np.pi)
            half_width = math.ceil(half_angle_x / pixel_angle)
            half_height = math.ceil(half_angle_y / pixel_angle)
            for dx in range(-half_width, half_width):
                for dy in range(-half_height, half_height):
                    x = center_x + dx
                    y = center_y + dy
                    if x < 0 or x >= img_width or y < 0 or y >= img_height:
                        continue
                    if depth_img[y, x] < distance:
                        return True
            return False
        elif action == 4:
            height_map = np.zeros_like(depth_img)
            for y in range(img_height):
                angle_y_tan = np.tan(abs(y - center_y) * pixel_angle * (np.pi / 180))
                height_map[y] = angle_y_tan * depth_img[y]
            half_angle_x = np.arctan(drone_width / (2 * distance)) * (180 / np.pi)
            half_angle_y = np.arctan(drone_height / (2 * distance)) * (180 / np.pi)
            half_width = math.ceil(half_angle_x / pixel_angle)
            half_width = 10
            height = math.ceil(img_height * 0.05)
            gradient_y = np.gradient(height_map, axis=0)
            gradient_threshold = 0.02
            for dx in range(-half_width, half_width):
                x = center_x + dx
                for dy in range(0, height):
                    y = dy
                    if x < 0 or x >= img_width or y < 0 or y >= img_height:
                        continue
                    gradient = abs(gradient_y[y, x])
                    if height_map[y, x] < distance and gradient <= gradient_threshold:
                        return True
            return False
        elif action == 5:
            height_map = np.zeros_like(depth_img)
            for y in range(img_height):
                angle_y_tan = np.tan(abs(y - center_y) * pixel_angle * (np.pi / 180))
                height_map[y] = angle_y_tan * depth_img[y]
            half_angle_x = np.arctan(drone_width / (2 * distance)) * (180 / np.pi)
            half_angle_y = np.arctan(drone_height / (2 * distance)) * (180 / np.pi)
            half_width = math.ceil(half_angle_x / pixel_angle)
            half_width = 10
            height = math.ceil(img_height * 0.05)
            gradient_y = np.gradient(height_map, axis=0)
            gradient_threshold = 0.02
            for dx in range(-half_width, half_width):
                x = center_x + dx
                for dy in range(-height, 0):
                    y = img_height + dy
                    if x < 0 or x >= img_width or y < 0 or y >= img_height:
                        continue
                    gradient = abs(gradient_y[y, x])
                    if height_map[y, x] < distance and gradient <= gradient_threshold:
                        return True
            return False
        else:
            return False
    def check_collision(self, depth_img):
        attention = ""
        if self._check_collision(depth_img, 1):
            attention += "MOVE_FORWARD will collide with objects. "
        if self._check_collision(depth_img, 4, distance=2.2):
            attention += "ASCEND will collide with objects. "
        if self._check_collision(depth_img, 5, distance=2.2):
            attention += "DESCEND will collide with objects. "
        if len(attention) == 0:
            attention = "None"
        return attention
    
    # 如果做了XX, 可能看到..., 对完成Subgoal是否合适
    def subgoal_state(self, log_dir=None, retry=0):
        subgoal = context.current_subgoal
        landmark = context.current_instruction['landmark'].copy()
        next_subgoal = context.next_subgoal
        start_time = time.time()
        if landmark is not None:
            landmarks = ' | '.join(landmark)
        else:
            landmarks = 'None'
        if len(context.reference_imagination) <= context.instruction_index:
            context.reference_imagination.append([])
        if len(context.reference_imagination[context.instruction_index]) > context.subgoal_index:
            if log_dir is not None:
                file_name = f"subgoal_state_pre_generated.txt"
                with open(os.path.join(log_dir, file_name), 'w+') as f:
                    f.write(self.model_name)
                    f.write("\n---\n")
                    f.write(context.reference_imagination[context.instruction_index][context.subgoal_index])
            return context.reference_imagination[context.instruction_index][context.subgoal_index]
        prompt = """[COGNITIVE ROLE]: Imagination

[TASK]: You are a drone performing navigation task, trying to achieve <subgoal>. Now please imagine the final VISUAL state when you have achieved the given <subgoal>. First-person View (The drone's camera is facing forward).

[INPUT]: 
<Current Subgoal>
<Landmarks>
<Next Subgoal>

[NOTE]:
1. Your imagination should be a very concise sentence ONLY describing the visual state of <Landmarks> after you achieved the <Current Subgoal>, including:
  - The location and distance(in meters) of <Landmarks>. Prefer SPATIAL terms: LEFT/RIGHT/CENTER, etc.
2. Consider the <Next Subgoal> to infer the final state of <Current Subgoal>.
3. You can only see a small window: a little to the left, right, up, and down (about 45° each way).
   - If something ends up underneath you, behind you, or too high/low (outside that window), do NOT mention it.
   - Example (general): After moving above a landmark, it is now below you and out of view.

[OUTPUT CONSTRAINTS]:
Strictly formatted as a JSON code block without any explanatory text.

[EXPECTED OUTPUT]:
```json
{{
  "state": "When I have finished the <subgoal>, I might see ...(the location and distance of the <Landmarks>)."
}}
```

### Input ###
<Current Subgoal>:
{subgoal}

<Landmarks>:
{landmark}

<Next Subgoal>:
{next_subgoal}"""
        prompt = prompt.format(subgoal=subgoal, landmark=landmarks, next_subgoal=next_subgoal)
        response_raw, tokens = self.llm.request_with_token_count(prompt, model_name=self.model_name)
        response = re.findall(r"```json(?:\w+)?\n(.*?)```", response_raw, re.DOTALL | re.IGNORECASE)
        if len(response) == 0:
            try: 
                response = json_repair.loads(response_raw)
                state = response['state']
            except Exception as e:
                response = response_raw
                state = response
        else:
            response = json_repair.loads(response[-1])
            state = response['state']
        if log_dir is not None:
            end_time = time.time()
            file_name = f"subgoal_state_{retry}.txt"
            with open(os.path.join(log_dir, file_name), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(response_raw)
                f.write("\n---\n")
                f.write(str(state))
                f.write("\n---\n")
                prompt_tokens, total_tokens, completion_tokens = tokens
                f.write(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, time: {int((end_time - start_time)*1000)}")
        context.reference_imagination[context.instruction_index].append(state)
        return state
    
class ProblemSolving:
    def __init__(self, global_config: CognitiveAgentConfig, attention:Attention = None):
        self.global_config = global_config
        config = global_config.problem_solving
        self.llm = LLMWrapper()
        self.model_name = config.model
        self.attention = attention
    def split(self, instructions: str, log_dir=None):
        start_time = time.time()
        prompt = """[COGNITIVE ROLE]: Problem-Solving

[General TASK]: You are an Instruction Manager for UAV Navigation. You have the following two tasks:
1. SENTENCE SEGMENTATION
- Split input text into individual sentences using periods as separators
- Preserve original wording including leading conjunctions (e.g., "and...")
- Maintain original capitalization and spacing

2. LANDMARK EXTRACTION
- Identify ALL navigational landmarks (physical objects/locations)
- Capture full noun phrases following prepositions: to/at/near/above/before
- Pay special attention to landmarks after temporal prepositions (e.g., "after the building")
- Retain modifiers: "small building", "shop entrance", etc.

[NOTE]:
- Verify period placement for accurate segmentation
- Strictly detect temporal clauses and adjust execution order
- Include ALL landmarks per sentence (1-3 typical)
- Ensure landmarks are bound to corresponding actions in complex sentences
- NEVER modify original wording in sub-instructions
- Eliminate all JSON syntax errors

[OUTPUT CONSTRAINTS]:
- Strictly formatted as a JSON code block without any explanatory text
- Always use arrays even for single items

[EXPECTED OUTPUT]:
```json
[
  {{
    "sub-instruction": "...",
    "landmark": ["LANDMARK1",...]
  }},
  {{
    "sub-instruction": "...",
    "landmark": ["LANDMARK1","LANDMARK2",...]
  }}
]
```

### Input ###
<Instructions>:
{instruction}"""
        prompt = prompt.format(instruction=instructions[0])
        response, tokens = self.llm.request_with_token_count(prompt, model_name=self.model_name)
        splited_instructions = re.findall(r"```json(?:\w+)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if len(splited_instructions) == 0:
            splited_instructions = [re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)]
        if log_dir is not None:
            end_time = time.time()
            with open(os.path.join(log_dir, 'extract_landmarks.txt'), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(response)
                f.write("\n---\n")
                f.write(splited_instructions[-1])
                f.write("\n---\n")
                prompt_tokens, total_tokens, completion_tokens = tokens
                f.write(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, time: {int((end_time - start_time)*1000)}")
        splited_instructions = json_repair.loads(splited_instructions[-1])
        return splited_instructions
    def split_subgoal(self, log_dir=None):
        instruction = context.current_instruction['sub-instruction']
        observation = context.current_observation
        start_time = time.time()
        prompt = """[COGNITIVE ROLE]: Problem-Solving

[General TASK]: You are a UAV Navigation Strategist. Perform means-end analysis to decompose the <current instruction> into sequential subgoals.

[Input]:
<Current Instruction>
<Current Observation>

[Detailed Task]: SUBGOAL EXTRACTION
  - Identify all embedded subgoals/intermediate objectives within the <Current Instruction>
  - Extract complete phrases denoting actions/steps (e.g., "turn left at intersection")
  - Sequence subgoals by PHYSICAL EXECUTION ORDER, prioritizing temporal logic over sentence structure (Example: "Do A when B" -> ["B", "A"])
  - Replace pronouns with explicit referents (e.g., "... A and ... it" -> ["... A", "... A"])
  - Preserve descriptive modifiers/contextual details.
  - Cross-reference <Current Observation> to ensure executable subgoal generation

[WARNING]:
  - Carefully follow <Current Instruction>, DO NOT involve extra descriptive words before <landmarks>, DO NOT add extra information not mentioned.

[OUTPUT CONSTRAINTS]:
- Strictly formatted as a JSON code block without any explanatory text
- Always use arrays even for single items

[EXPECTED OUTPUT]:
```json
["SUBGOAL1", "SUBGOAL2"]
```

### Input ###
<Current Instruction>:
{instruction}

<Current Observation>:
{observation}"""
        prompt = prompt.format(instruction=instruction, observation=observation)
        response, tokens = self.llm.request_with_token_count(prompt, model_name=self.model_name)
        splited_subgoals = re.findall(r"```json(?:\w+)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if len(splited_subgoals) == 0:
            splited_subgoals = [re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)]
        if log_dir is not None:
            with open(os.path.join(log_dir, 'extract_subgoals.txt'), 'w+') as f:
                end_time = time.time()
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(response)
                f.write("\n---\n")
                f.write(splited_subgoals[-1])
                f.write("\n---\n")
                prompt_tokens, total_tokens, completion_tokens = tokens
                f.write(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, time: {int((end_time - start_time)*1000)}")
        splited_subgoals = json_repair.loads(splited_subgoals[-1])
        if len(splited_subgoals) == 0:
            splited_subgoals = [instruction]
        return splited_subgoals

# judger
class Reasoning:
    def __init__(self, global_config: CognitiveAgentConfig, attention:Attention = None, imagery: Imagery = None, problem_solving: ProblemSolving = None, perception: Perception = None):
        self.global_config = global_config
        config = global_config.reasoning
        self.llm = LLMWrapper()
        self.model_name = config.model
        self.imagery = imagery
        self.attention = attention
        self.problem_solving = problem_solving
        self.perception = perception
        self.instruction_steps = 0
        self.subgoal_steps = {}
        self.subgoals = None
    def subgoal_achieved(self, log_dir=None, retry=0):
        subgoal = context.current_subgoal
        next_subgoal = context.next_subgoal
        scene = context.current_observation
        history = context.get_current_subgoal_memory()
        start_time = time.time()
        state = self.imagery.subgoal_state(log_dir=log_dir, retry=retry)
        prompt = """[COGNITIVE ROLE]: Reasoning

[TASK]: You are a drone navigation analysis expert. Your task is to estimate whether the subgoal has been achieved or not. 

[INPUT]:
<Current Subgoal>
<Subgoal Memory>: What you have ALREADY DONE and OBSERVED for <Current Subgoal>.
<Current Observation>
<Reference Imagination>: The blind imagined reference VISUAL state when you have achieved the <Subgoal>.
<Next Subgoal>: The next subgoal you are trying to transition to

[NOTE]:
1. To make your decision, analyze the inputs as follows:
- Check the <Current Observation>, mainly focus on the status of Landmarks, and then decide if it is time to switch from the <Current Subgoal> to <Next Subgoal>.
- Review the <Subgoal Memory> to check whether the actions taken align with the <Current Subgoal>.
- <Reference Imagination> is a blind guess, take it for reference occasionally. 
2. If the subgoal says **turn left/right** without a specified degree or reference, it means a large turn, usually > 45 degrees( > 3 times).
3. If the <Current Observation> is not matched with the <Reference Imagination>, but still make sense as both initial state of the <Next Subgoal> and final state of the <Current Subgoal>, then it might also achieved.
4. If there are multiple(>15) steps in the <Subgoal Memory>, you should check whether you have achieved the <Current Subgoal> before the last step.

[ADDITIONAL REQUIREMENTS]:
1. [Action Compliance Verification] For each subgoal evaluation, you MUST verify:
  - Every action SPECIFIED in the subgoal description has been executed (cross-validate with <Subgoal Memory>)

[OUTPUT CONSTRAINTS]:
1. Strictly formatted as a JSON code block without any explanatory text.
2. Please also provide a brief explanation involving your evidence from [INPUT].

[EXPECTED OUTPUT]:
```json
{{
  "achieved": <true | false>,
  "reason": "...(brief explanation)"
}}
```

### Input ###
<Current Subgoal>:
{subgoal}

<Subgoal Memory>:
{history}

<Current Observation>:
{scene}

<Reference Imagination>:
{state}

<Next Subgoal>:
{next_subgoal}"""
        prompt = prompt.format(next_subgoal=next_subgoal, subgoal=subgoal, scene=scene, history=history, state=state)
        response_raw, tokens = self.llm.request_with_token_count(prompt, model_name=self.model_name)
        response = re.findall(r"```json(?:\w+)?\n(.*?)```", response_raw, re.DOTALL | re.IGNORECASE)
        if len(response) == 0:
            try: 
                response = json_repair.loads(response_raw)
                status = response['achieved']
            except Exception as e:
                response = response_raw
                status = False
        else:
            response = json_repair.loads(response[-1])
            try:
                status = response['achieved']
            except:
                print(f"Error in response: {response}")
                status = False
        if log_dir is not None:
            end_time = time.time()
            with open(os.path.join(log_dir, f'subgoal_achieved_{retry}.txt'), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(response_raw)
                f.write("\n---\n")
                f.write(str(response))
                f.write("\n---\n")
                prompt_tokens, total_tokens, completion_tokens = tokens
                f.write(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, time: {int((end_time - start_time)*1000)}")
        return status
    def instruction_completed(self, history: Memory, log_dir=None, retry=0):
        current_instruction = context.current_instruction
        next_instruction = context.next_instruction
        scene = context.current_observation
        current_instruction_text = current_instruction['sub-instruction']
        next_instruction_text = next_instruction['sub-instruction'] if next_instruction is not None else None
        instruction_steps = 0
        if context.instruction_index < len(context.step_memory):
            for item in context.step_memory[context.instruction_index]:
                instruction_steps += len(item)
        # if self.instruction_finished(current_instruction=current_instruction, next_instruction=next_instruction_text, scene=scene, history=history.history, log_dir=log_dir, retry=retry):
        noise = random.normalvariate(instruction_steps, 5)
        # noise = 0
        if noise > 30:
            context.instruction_index += 1
            context.current_instruction = context.sub_instructions[context.instruction_index] if context.instruction_index < len(context.sub_instructions) else None
            context.next_instruction = context.sub_instructions[context.instruction_index + 1] if context.instruction_index + 1 < len(context.sub_instructions) else None
            if next_instruction is not None:
                context.subgoals.append(self.problem_solving.split_subgoal(log_dir=log_dir))
                context.subgoal_index = 0
                context.current_subgoal = context.subgoals[context.instruction_index][context.subgoal_index]
                context.next_subgoal = context.subgoals[context.instruction_index][context.subgoal_index + 1] if context.subgoal_index + 1 < len(context.subgoals[context.instruction_index]) else context.next_instruction['sub-instruction'] if context.next_instruction is not None else None
                return True, context.subgoals[context.instruction_index][context.subgoal_index], 1
            else:
                return True, None, 2
        else:
            if len(context.subgoals) <= context.instruction_index:
                context.subgoals.append(self.problem_solving.split_subgoal(log_dir=log_dir))
                context.subgoal_index = 0
                context.current_subgoal = context.subgoals[context.instruction_index][context.subgoal_index]
                context.next_subgoal = context.subgoals[context.instruction_index][context.subgoal_index + 1] if context.subgoal_index + 1 < len(context.subgoals[context.instruction_index]) else context.next_instruction['sub-instruction'] if context.next_instruction is not None else None
            subgoals = context.subgoals[context.instruction_index]
            if self.subgoal_achieved(log_dir=log_dir, retry=retry):
                history.update_instruction_memory(log_dir=log_dir)
                context.subgoal_index += 1
                if context.subgoal_index < len(subgoals):
                    context.current_subgoal = subgoals[context.subgoal_index]
                    context.next_subgoal = subgoals[context.subgoal_index + 1] if context.subgoal_index + 1 < len(subgoals) else next_instruction_text if next_instruction is not None else None
                    return False, context.current_subgoal, 3
                else:
                    context.instruction_index += 1
                    context.current_instruction = context.sub_instructions[context.instruction_index] if context.instruction_index < len(context.sub_instructions) else None
                    context.next_instruction = context.sub_instructions[context.instruction_index + 1] if context.instruction_index + 1 < len(context.sub_instructions) else None
                    if next_instruction is not None:
                        context.subgoals.append(self.problem_solving.split_subgoal(log_dir=log_dir))
                        context.subgoal_index = 0
                        context.current_subgoal = context.subgoals[context.instruction_index][context.subgoal_index]
                        context.next_subgoal = context.subgoals[context.instruction_index][context.subgoal_index + 1] if context.subgoal_index + 1 < len(context.subgoals[context.instruction_index]) else context.next_instruction['sub-instruction'] if context.next_instruction is not None else None
                        return True, context.current_subgoal, 4
                    else:
                        return True, None, 5
            else:
                return False, subgoals[context.subgoal_index], 6
# planner
class DecisionMaking:
    def __init__(self, global_config: CognitiveAgentConfig, attention:Attention = None, memory:Memory = None):
        self.global_config = global_config
        config = global_config.decision_making
        self.llm = LLMWrapper()
        self.model_name = config.model
        self.attention = attention
        self.memory = memory
    
    def plan(self, log_dir=None, step=0, retry=0):
        current_instruction = context.current_instruction
        current_instruction_text = current_instruction['sub-instruction']
        next_instruction = context.next_instruction
        next_instruction_text = next_instruction['sub-instruction'] if next_instruction is not None else None
        collision_warning = context.collision_warning
        subgoal = context.current_subgoal
        scene = context.current_observation
        start_time = time.time()
        actions = """1: MOVE_FORWARD (5 meters)
2: TURN_LEFT (15 degrees)
3: TURN_RIGHT (15 degrees)
4: ASCEND (2 meters)
5: DESCEND (2 meters)
6: MOVE_LEFT (5 meters)
7: MOVE_RIGHT (5 meters)"""
        if next_instruction is None:
            actions = """0: TASK_FINISH
1: MOVE_FORWARD (5 meters)
2: TURN_LEFT (15 degrees)
3: TURN_RIGHT (15 degrees)
4: ASCEND (2 meters)
5: DESCEND (2 meters)
6: MOVE_LEFT (5 meters)
7: MOVE_RIGHT (5 meters)"""
        prompt = """[COGNITIVE ROLE]: Decision-Making

[TASK]: You are an embodied drone that navigates in the real world. Your task is to CHOOSE the most reasonable action to be executed next from the VALID ACTIONS, so that you could achieve the subgoal.

[INPUT]: 
<Instruction Memory>: What you have ALREADY DONE and OBSERVED for the current instruction.
<Current Observation>
<Collision Warning>
<Current Instruction>
<Current Subgoal>: The subgoal of the current instruction you are working on.
<Valid Actions>

[NOTE]:
1. To make your decision, analyze the inputs as follows:
  - Understand the <Current Subgoal>, find out the actions aligned with it.
  - Consider the <Collision Warning> to avoid collisions.
2. **Probability Rules**:
  - Output probabilities for **ALL Valid Actions**
  - Higher probability = stronger preference
3. Carefully check the <Instruction Memory>: If over multiple steps you consistently observe scenes that no visible landmarks - immediately select the action 0

[OUTPUT CONSTRAINTS]:
1. Strictly formatted as a JSON code block without any explanatory text.
2. The [OUTPUT] JSON must include the following information:
  - "thought": why you choose this action, considering the <Current Subgoal>, <Current Observation>, <Instruction Memory>, <Current Instruction> and <Collision Warning>.
  - "probabilities": probability distribution over the valid action list.
  - "selected_action": Explicitly select the action with highest probability. Note that only output the action number.

### INPUT
<Instruction Memory>:
{history}

<Current Observation>:
{scene}

<Collision Warning>:
**{collision_warning}**

<Current Instruction>:
{current_instruction}

<Current Subgoal>:
{subgoal}

<Valid Actions>:
{actions}"""
        
        history = self.memory.history
        prompt = prompt.format(current_instruction=current_instruction_text, scene=scene, collision_warning=collision_warning, history=history, subgoal=subgoal, actions=actions)
        responses_raw, tokens = self.llm.request_with_token_count(prompt, model_name=self.model_name)
        responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
        if len(responses) == 0:
            try: 
                response = json_repair.loads(responses_raw)
                action = response['selected_action']
            except Exception as e:
                try:
                    responses = re.findall(r"\{[\s\S]*\}", responses_raw, re.DOTALL | re.IGNORECASE)
                    if len(responses) != 0:
                        response = json_repair.loads("{" + responses[-1] + "}")
                        action = response['selected_action']
                    else:
                        responses = responses_raw
                        print(f"Cannot parse response: {responses_raw}")
                        action = 0
                except Exception as ee:
                    responses = responses_raw
                    print(f"Cannot parse response: {responses_raw}")
                    action = 0
        else:
            try:
                response = json_repair.loads(responses[-1])
                action = response['selected_action']
            except:
                action = 0
        try:
            action = int(action)
            if action > 7 or action < 0:
                action = 0
        except:
            action = 1
        
        if log_dir is not None:
            end_time = time.time()
            file_name = f"plan_{retry}.txt"
            with open(os.path.join(log_dir, file_name), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(responses_raw)
                f.write("\n---\n")
                f.write(str(action))
                f.write("\n---\n")
                prompt_tokens, total_tokens, completion_tokens = tokens
                f.write(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, time: {int((end_time - start_time)*1000)}")
        return action

class CognitiveAgent:
    def __init__(self, global_config: CognitiveAgentConfig):
        self.global_config = global_config
        self.attention = Attention(global_config)
        self.memory = Memory(global_config, attention=self.attention)
        self.imagery = Imagery(global_config, attention=self.attention)
        self.problem_solving = ProblemSolving(global_config, attention=self.attention)
        self.decision_making = DecisionMaking(global_config, attention=self.attention, memory=self.memory)
        self.perception = Perception(global_config, attention=self.attention)
        self.reasoning = Reasoning(global_config, attention=self.attention, imagery=self.imagery, problem_solving=self.problem_solving, perception=self.perception)
        self.vlm_model = global_config.perception.vlm_model
        self.detector = global_config.perception.detector
    def eval(self):
        pass
    def preprocess(self, observations, log_dir=None):
        context.instruction_text = observations['instruction']
        context.sub_instructions = []
        context.sub_instructions.extend(self.problem_solving.split(context.instruction_text, log_dir=log_dir))
        context.episode_id = observations['episode_id']
        context.trajectory_id = observations['trajectory_id']
        context.scenes_id = observations['scene_id']
        context.macro_step = 0
        context.actions = []
        context.collision_warning = ""
        context.attention_questions = []
        context.current_observation = {}
        context.current_instruction = context.sub_instructions[0]
        context.next_instruction = context.sub_instructions[1] if len(context.sub_instructions) > 1 else None
        context.instruction_index = 0
        context.subgoals = []
        context.current_subgoal = None
        context.next_subgoal = None
        context.subgoal_index = 0
        context.reference_imagination = []
        context.subgoal_memory = []
        context.step_memory = []
        context.log_dir_base = str(log_dir)
        context.log_dir = str(log_dir)
    
    def act(self, observations, step = 0):
        log_dir = context.log_dir_base
        if log_dir is not None:
            log_dir = os.path.join(log_dir, f'step_{step}')
            context.log_dir = log_dir
            os.makedirs(log_dir, exist_ok=True)
            img_path = os.path.join(log_dir, f'{step}.png')
        else:
            img_path = None
        context.macro_step = step
        
        actions = []
        finisheds = []
        instructions = observations['instruction']
        rgbs = observations['rgb']
        depths = observations['depth']
        for i in range(len(instructions)):
            instruction = instructions[i]
            rgb = rgbs[i]
            depth = depths[i]
            finished = False
            if log_dir is not None: 
                depth_unit8 = (depth*255).astype(np.uint8)
                cv2.imwrite(os.path.join(log_dir, f'{step}_depth.png'), depth_unit8)
            ############## PROCESS AND GET INSTRUCTIONS ############
            current_instruction = context.current_instruction
            current_instruction_text = current_instruction['sub-instruction']
            next_instruction = context.next_instruction
            
            ############ SCENE PERCEPTION plus check_collision
            scene = self.perception.get_scene(rgb=rgb, log_dir=log_dir, img_path=img_path)
            collision_warning = self.imagery.check_collision(depth * 100)
            context.collision_warning = collision_warning
            ###############################
            finished, subgoal, code = self.reasoning.instruction_completed(history=self.memory, log_dir=log_dir)
            self.memory.history = context.get_instruction_memory()
            self.memory.history = self.memory.history.strip()
            if log_dir is not None:
                with open(os.path.join(log_dir, 'instruction_completed.txt'), 'w+') as f:
                    f.write(str(self.reasoning.subgoal_steps))
                    f.write("\n---\n")
                    f.write(str(finished))
                    f.write("\n---\n")
                    f.write(str(subgoal))
                    f.write("\n---\n")
                    f.write(str(code))
            if finished:
                finished = True
                print(f'Instruction {context.instruction_index} finished')
                if context.current_instruction is None:
                    action = 0
                    context.actions.append(action)
                    finisheds.append(finished)
                    actions.append(action)
                    if log_dir is not None:
                        with open(os.path.join(log_dir, 'states.txt'), 'w') as f:
                            f.write(json.dumps(asdict(context)))
                    continue
                self.memory.clear()
                current_instruction = context.current_instruction
                current_instruction_text = current_instruction['sub-instruction']
                next_instruction = context.next_instruction
                print(f'Current Instruction: {current_instruction_text}')
                scene = self.perception.get_scene(rgb=rgb, reget=True, log_dir=log_dir)
            
            print(f'Subgoal: {subgoal}')
            action = self.decision_making.plan(log_dir=log_dir, step=step)
            self.memory.add_tuple(action=action, observation=scene)
            self.memory.generate_step_memory(log_dir=log_dir)
            self.memory.history = context.get_instruction_memory()
            self.memory.history = self.memory.history.strip()
            finisheds.append(finished)
            actions.append(action)
            context.actions.append(action)
            if log_dir is not None:
                with open(os.path.join(log_dir, 'states.txt'), 'w') as f:
                    f.write(json.dumps(asdict(context)))
        print(f'Action: {actions}')
        return actions, finisheds