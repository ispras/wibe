import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from ..base import BaseAttack
from wibench.typing import TorchImg


class StegastampInversion(BaseAttack):
    """Attack from `here <https://github.com/leiluk1/erasing-the-invisible-beige-box/blob/main/notebooks/stegastamp_attack.ipynb>`__.

    **TODO**:

        * check that this works the same as notebook
        * run with GPU tensors, see the following `link <https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py#L66>`__
        * convert from onnx to pytorch?
    """

    def __init__(self,
                 stegastamp_model_path: str,
                 device_id: int = 0,
                 ) -> None:
        super().__init__()

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        self.stegastamp_model = ort.InferenceSession(
            stegastamp_model_path,
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(device_id)}],
            sess_options=session_options,
        )

        self.secret_len = 100

    def encode(self, image: np.ndarray, secret: np.ndarray) -> np.ndarray:
        stegastamp, residual, decoded = self.stegastamp_model.run(output_names=None, input_feed={"image": image, "secret": secret})
        return stegastamp, residual

    def decode(self, img: np.ndarray) -> np.ndarray:
        dummy_secret = np.zeros((img.shape[0], self.secret_len), dtype=np.float32)  # not used by the model
        stegastamp, residual, decoded = self.stegastamp_model.run(
            output_names=None,
            input_feed={"image": img, "secret": dummy_secret},
        )
        return decoded

    def __call__(self, img: TorchImg) -> TorchImg:
        img = img.unsqueeze(0)
        b, c, h, w = img.shape
        img = F.interpolate(img, size=(400, 400), mode="bicubic", antialias=True)  # [b,c,400,400]
        img_np = img.permute(0, 2, 3, 1).detach().cpu().numpy()  # [b,400,400,3]

        watermarks = self.decode(img_np)
        inverted_mask = 1 - watermarks
        attacked_img, _ = self.encode(img_np, inverted_mask)

        attacked_img = torch.tensor(attacked_img).permute(0, 3, 1, 2).to(img.device)
        attacked_img = F.interpolate(attacked_img, size=(h, w), mode="bicubic")
        return attacked_img.squeeze(0)
