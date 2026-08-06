# Changelog

## Version 0.4.0 - 06.08.26

WARP paper release

This release significantly extends the WIBE benchmark with new watermarking methods, removal and forgery attacks, metrics, datasets, and a new dependency-management system for running components with conflicting requirements.

### Added

- Attacks: Instagram filters, NRP, MPRNet, UniEditFlux, DISCO, TrustMarkRM, DiffPure, RealESRGAN, support for any attack combination and embedding watermark as an attack
- Methods: VINE, SepMark, Rosteals
- Metrics: WER, Empirical TPR@xFPR
- MSCOCO with captions

### Features

- New wibench-venv command-line utility.
- Profile-based virtual environments under profiles/.
- Automatic grouping of compatible component requirements.
- Smart dependencies management with uv
- Logging
- Cuda visible devices for post pipeline metrics fix
- Some more bugfixes

## Version 0.3.0 - 13.02.26

Added:

- Post-hoc watermarking methods: Chunkyseal, Pixelseal, Videoseal, Robust-Wide, PIMoG, MaskWM, FIN, invismark, mbrs
- Syncseal geometric synchronization method
- built-in watermarking methods: Ring-ID, METR, Gaussian Shading, MaXsive
- FID and DreamSim image quality metrics
- Attacks on watermarks: Averaging, Frequency Masking, blur-deblur, Image editing, LIIF, SEMAttack, WMForger

Features:

- Dynamic resource files downloading
- Smart module importing (support for simultaneous usage of several algorithms)
- post_pipeline stages (only FID metric for now)

## Version 0.2.1 - 20.01.26

Minor changes and fixes

## Version 0.2.0 - 12.08.25

Release presented at ASE 2025. Added key features of Wibe:

- 15 watermarking methods
- 23 attacks on watermarks
- Image quality and extraction robustness metrics

## Version 0.1.0 - 01.07.25

Old version for inner use
