# WIBE: Watermarks for generated Images – Benchmarking & Evaluation

![Documentation Status](https://readthedocs.org/projects/example-sphinx-basic/badge/?version=latest)

**WIBE** is a modular and extensible framework for automated testing of invisible image watermarking methods under various attack scenarios.
The system is designed to support research and development of robust watermarking techniques by enabling systematic evaluation
through a customizable processing pipeline.

The system architecture consists of a sequence of processing configurable stages.

![`WIBE schema`](docs/imgs/wibe_schema.png)

## Key features

* Modularity and extensibility through a plugin-based architecture
* Reproducibility ensured by YAML-configured experiments
* Usability with a simple command-line interface
* Flexible persistence through multiple storage backends, including files and ClickHouse database
* Transparency via real-time visual feedback
* Scalability to run experiments on clusters

## Quick start

To assess implemented watermarking algorithms and attacks on watermarks, follow the step-by-step procedure below.

1. Clone the repository and navigate to its directory (all subsequent commands should be run from this location):

```console
git clone https://github.com/ispras/wibe.git
```

2. Update the submodules:

```console
git submodule update --init --recursive
```

3. Create and activate a virtual environment (the exact command varies slightly between OSes – you know how to do this):

```console
python -m venv venv
```

4. Download the pre-trained model weights:

```console
(venv) python download_models.py
```

5. Install the dependencies:

```console
(venv) python install_requirements.py
```

6. Set the **HF_TOKEN** environment variable with your **HuggingFace** [token](https://huggingface.co/settings/tokens) (see [HuggingFace Authentication Setup](https://ispras-wibe.readthedocs.io/en/latest/quick_start.html#huggingface-authentication-setup) for details), then authenticate:

```console
(venv) python huggingface_login.py
```

7. All set! Specify the path to your `сonfiguration file` as a required parameter:

```console
(venv) python -m wibench --config configs/trustmark_demo.yml -d
```

8. Upon completion of computations, you can view watermarked images and explore interactive charts for different combinations of watermarking algorithms, attacks, and computed performance metrics.

Below, from left to right, are the original, watermarked with [StegaStamp](https://www.matthewtancik.com/stegastamp), and attacked by [FLUX Regeneration](https://github.com/leiluk1/erasing-the-invisible-beige-box/blob/main/notebooks/treering_attack.ipynb) images.

![`Original, watermarked, and attacked images`](docs/imgs/original_watermarked_attacked.png)

And here are the same as above, the original and watermarked images, as well as their difference.

![`Original and watermarked images, and their difference`](docs/imgs/original_watermarked_difference.png)

To explore interactive wind rose chart with average `TPR@0.1%FPR` for all algorithms and attacks evaluated so far, run the following command:

```console
(venv) python make_plots.py --results_dir path_to_results_directory
```

Below is an average `TPR@0.1%FPR` chart for 7 algorithms under different types of attacks (evaluated on 300 images from the [DiffusionDB](https://poloclub.githubithub.io/diffusiondb/) dataset).

![`Average TPR@0.1%FPR for 7 algorithms`](docs/imgs/tpr_0.1_fpr_avg.png)

## Documentation

See the full documentation [here](https://ispras-wibe.readthedocs.io/en/latest/index.html).

## Tutorial video

Watch our video tutorial [here](https://youtu.be/31kiJ8G2NG8).
