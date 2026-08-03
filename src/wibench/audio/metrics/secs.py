from typing import Optional, Any
from pathlib import Path
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from wibench.download import requires_download
from wibench.pipeline_type import PipelineType
from wibench.common.metrics import PostEmbedMetric
from wibench.typing import TorchAudio
from wibench.audio.metrics.utils import align_pair

URL = "https://nextcloud.ispras.ru/index.php/s/YQFGkgaBqJ4pz6q"
NAME = "SECS"
REQUIRED_FILES = [
    "spkrec-ecapa-voxceleb/classifier.ckpt",
    "spkrec-ecapa-voxceleb/embedding_model.ckpt",
    "spkrec-ecapa-voxceleb/hyperparams.yaml",
    "spkrec-ecapa-voxceleb/label_encoder.ckpt",
    "spkrec-ecapa-voxceleb/mean_var_norm_emb.ckpt",
]

DEFAULT_CACHE_DIR = "./model_files/SECS"


@requires_download(URL, NAME, REQUIRED_FILES)
class SECS(PostEmbedMetric):
    """
    Speaker Encoder Cosine Similarity (SECS).

    The implementation is taken from `speechbrain <https://pypi.org/project/speechbrain/>`_ library.
    """

    pipeline_type = PipelineType.AUDIO

    def __init__(
        self,
        target_rate: int = 16000,
        device: Optional[str] = None,
    ):
        """
        Initialization Parameters
        -------------------------
        target_rate: int
            Target sampling rate (frequency).
        device: str (optional)
            Metric computation device.
        """
        self.target_rate = target_rate
        self.device = device or "cpu"

        from speechbrain.inference.speaker import EncoderClassifier

        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=Path(DEFAULT_CACHE_DIR) / "spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )

    def _prepare(
        self,
        audio: TorchAudio,
    ) -> torch.Tensor:
        wav = audio.data.to(torch.float32)

        if audio.rate != self.target_rate:
            wav = Resample(
                orig_freq=audio.rate,
                new_freq=self.target_rate,
            )(wav)

        return wav.to(self.device)

    def score(
        self,
        ref_audio: TorchAudio,
        deg_audio: TorchAudio,
    ) -> float:
        with torch.inference_mode():
            ref = self._prepare(ref_audio)
            deg = self._prepare(deg_audio)

            if ref.shape[0] != deg.shape[0]:
                raise ValueError(
                    f"Different number of channels: "
                    f"{ref.shape[0]} != {deg.shape[0]}"
                )

            ref, deg = align_pair(
                ref,
                deg,
                mono=False,
            )

            ref_emb = self.encoder.encode_batch(ref).squeeze(1)
            deg_emb = self.encoder.encode_batch(deg).squeeze(1)

            scores = F.cosine_similarity(
                ref_emb,
                deg_emb,
                dim=-1,
            )

            return float(scores.mean().item())

    def __call__(
        self,
        audio1: TorchAudio,
        audio2: TorchAudio,
        watermark_data: Any,
    ) -> float:
        audio1 = TorchAudio(*audio1)
        audio2 = TorchAudio(*audio2)

        return self.score(
            audio1,
            audio2,
        )
