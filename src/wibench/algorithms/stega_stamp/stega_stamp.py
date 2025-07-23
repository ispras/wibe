import onnxruntime as ort
import numpy as np
import cv2


class StegaStamp:
    def __init__(self, model_path: str, wm_length: int = 100, width: int = 400, height: int = 400, alpha: float = 1):
        self._model = ort.InferenceSession(model_path)
        self.wm_length = wm_length
        self.width = width
        self.height = height
        self.alpha = alpha

    def encode(
        self, 
        image: np.ndarray, 
        message: np.ndarray,
    ) -> np.ndarray:
        orig_height, orig_width = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        float_img = np.divide(resized_img, 255, dtype=np.float32)
        batched_img = float_img[np.newaxis, ...]
        batched_message = message[np.newaxis, ...].astype(np.float32)

        wm_img, _ = self._model.run(
            output_names=['stegastamp', 'residual'],
            input_feed={'image': batched_img, 'secret': batched_message}
        )
        result = np.round(np.clip(wm_img[0], 0, 1) * 255).astype(np.int16)
        diff = np.round((result - resized_img) * self.alpha).astype(np.int16)
        min_val = diff.min()
        diff_resized = cv2.resize((diff - min_val).astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        marked_rgb = img_rgb + diff_resized.astype(np.int16) + min_val
        marked_uint = np.clip(marked_rgb, 0, 255).astype(np.uint8)
        return cv2.cvtColor(marked_uint, cv2.COLOR_RGB2BGR)

    def decode(self, image: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        float_img = np.divide(resized_img, 255, dtype=np.float32)
        batched_img = float_img[np.newaxis, ...]

        secret = self._model.run(
            output_names=['decoded'],
            input_feed={'image': batched_img, 'secret': np.zeros((1, self.wm_length), dtype=np.float32)}
        )[0]

        return (secret > 0.5).astype(int)