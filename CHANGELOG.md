# Changelog

## Version 0.4.0 -TBA

Added:

- Attacks: NRP, MPRNet, UniEditFlux, DISCO and support for any attack combination

Features:

- Smart dependencies managements with uv
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

## Version 0.2.0 - 12.08.26

Release presented at ASE 2025. Added key features of Wibe:

- 15 watermarking methods
- 23 attacks on watermarks
- Image quality and extraction robustness metrics

## Version 0.1.0 - 01.07.25

Old version for inner use
