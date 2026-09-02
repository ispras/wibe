from .signal import SignInversion, Resampling, Requantization, Gain, Clipping, Filter, WhiteNoise, PinkNoise
from .compression import Mpeg, AAC, Opus
from .acoustic import Echo, Reverb
from .effect import PitchShift, DynamicRangeCompressor
from .enhancement import WienerFilter
from .speech_enhancement import GTCRN, MetricGANPlus
from .malicious import Vocos
from .desynchronization import TimeStretch, Speed, InvertedTimeStretch, FlipSamples, ZeroCrossInserts, ReplacementAttack, Crop, FrameDropout
