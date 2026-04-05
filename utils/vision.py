from PIL import Image
from typing import Optional
from numpy.typing import NDArray
import numpy as np
import cv2
from src.common.vlm_wrapper import VLMWrapper, LLAMA3V


class Frame():
    def __init__(self, image: Image.Image | NDArray[np.uint8]=None, depth: Optional[NDArray[np.int16]]=None):
        if image is None:
            self._image_buffer = np.zeros((352, 640, 3), dtype=np.uint8)
            self._image = Image.fromarray(self._image_buffer)
        if isinstance(image, np.ndarray):
            self._image_buffer = image
            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._image = Image.fromarray(im_rgb)
        elif isinstance(image, Image.Image):
            self._image = image
            self._image_buffer = np.array(image)
        self._depth = depth
    
    @property
    def image(self) -> Image.Image:
        return self._image
    
    @property
    def depth(self) -> Optional[NDArray[np.int16]]:
        return self._depth
    
    @image.setter
    def image(self, image: Image.Image):
        self._image = image
        self._image_buffer = np.array(image)

    @depth.setter
    def depth(self, depth: Optional[NDArray[np.int16]]):
        self._depth = depth

    @property
    def image_buffer(self) -> NDArray[np.uint8]:
        return self._image_buffer
    
    @image_buffer.setter
    def image_buffer(self, image_buffer: NDArray[np.uint8]):
        self._image_buffer = image_buffer
        self._image = Image.fromarray(image_buffer)
    
class VisionClient():
    def __init__(self, detector: str = 'vllm', vlm_model: str = LLAMA3V):
        self.detector = detector
        self.vlm_model = vlm_model
        self.vlm_client = VLMWrapper()
    
    def set_vlm(self, model_name: str):
        self.vlm_model = model_name
    
    def detect_capture_with_token_count(self, rgb, depth=None, prompt: str = None, save_path = None):
        frame = Frame(rgb, depth)
        return self.vlm_client.request_with_token_count(prompt, model_name=self.vlm_model, image=frame.image, save_path=save_path)