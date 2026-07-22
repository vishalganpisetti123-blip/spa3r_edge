<div align="center">

# 🔮 ***[Spae]()ing the Unseen***

### Spa3R: Predictive Spatial Field Modeling for 3D Visual Reasoning

[Haoyi Jiang](https://scholar.google.com/citations?user=_45BVtQAAAAJ)<sup>1</sup>,
Liu Liu<sup>2</sup>, Xinjie Wang<sup>2</sup>, Yonghao He<sup>3</sup>,<br>
Wei Sui<sup>3</sup>, Zhizhong Su<sup>2</sup>,
Wenyu Liu<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1</sup><br>
<sup>1</sup>Huazhong University of Science & Technology,
<sup>2</sup>Horizon Robotics,
<sup>3</sup>D-Robotics

[![arXiv](https://img.shields.io/badge/arXiv-2602.21186-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2602.21186)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/hustvl/Spa3-VLM)
[![License: MIT](https://img.shields.io/github/license/hustvl/Spa3R)](LICENSE)

</div>

## Installation

Please clone this project with `--recursive`.

```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

pip install submodules/vggt
pip install -e submodules/lmms-eval
```

## Data Preparation

### 1. Pre-training

We utilize a combination of large-scale indoor scene datasets: [ScanNet](http://www.scan-net.org/) and [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/).

### 2. Instruction Tuning

* **Video-centric VSI-Bench:** We fine-tune our model on the [VSI-590K](https://huggingface.co/datasets/nyu-visionx/VSI-590K) dataset.
* **Image-based benchmarks:** We use a composite training set aligned with [VG-LLM](https://github.com/LaVi-Lab/VG-LLM?tab=readme-ov-file#datasets). 

Our processed annotations are available [here](https://huggingface.co/datasets/hustvl/Spa3R-VLM). Please configure the local data and annotation paths in `data/__init__.py` before starting the training.

## Training

### 1. Spa3R Pre-training

To train the Predictive Spatial Field Modeling (PSFM) framework from scratch:

```
export PYTHONPATH=.
python scripts/train_spa3r.py
```

### 2. Spa3-VLM Instruction Tuning

Set the pre-trained Spa3R path in the script: `geometry_encoder_path=/path/to/spa3r.ckpt`

```
bash scripts/train_vlm_sft.sh
```

## Evaluation

We provide pre-trained weights for [Spa3R](https://huggingface.co/hustvl/Spa3-VLM/blob/main/spa3r.ckpt) and [Spa3-VLM on VSI-590K](https://huggingface.co/hustvl/Spa3-VLM/tree/main/spa3_vlm) on Hugging Face.

To evaluate Spa3-VLM on spatial reasoning benchmarks:

```
bash scripts/eval_vlm.sh
```

## Edge-Spa3R local simulation

A lightweight local edge/cloud split simulation is now available for prototyping the distributed architecture described in the project blueprint.

### Included scripts

- [export_encoder.py](export_encoder.py): exports a placeholder PyTorch encoder to an ONNX graph with dynamic view-count axes.
- [quantize_encoder.py](quantize_encoder.py): quantizes the exported encoder to INT8 with ONNX Runtime.
- [edge_node.py](edge_node.py): simulates an edge device with CPU affinity, ONNX Runtime inference, and latent transmission over TCP.
- [cloud_server.py](cloud_server.py): receives the spatial latent tensor and produces a lightweight cloud-side reasoning response.
- [benchmark/latency_bench.py](benchmark/latency_bench.py): benchmarks FP32 vs. INT8 inference and payload serialization overhead.

### Run locally

```bash
/Users/vishal/Documents/GitHub/Spa3R/.venv/bin/python export_encoder.py
/Users/vishal/Documents/GitHub/Spa3R/.venv/bin/python quantize_encoder.py
/Users/vishal/Documents/GitHub/Spa3R/.venv/bin/python cloud_server.py
/Users/vishal/Documents/GitHub/Spa3R/.venv/bin/python edge_node.py
/Users/vishal/Documents/GitHub/Spa3R/.venv/bin/python benchmark/latency_bench.py
```

> The current implementation is intentionally lightweight and uses a simulated edge process rather than physical Raspberry Pi hardware. The benchmarking output is therefore labeled as a software simulation, not a measured Pi 5 deployment.

## Citation

If you find our work helpful for your research, please consider starring this repository :star: and citing our work:

```bibtex
@article{Spa3R,
  title={Spa3R: Predictive Spatial Field Modeling for 3D Visual Reasoning},
  author={Haoyi Jiang and Liu Liu and Xinjie Wang and Yonghao He and Wei Sui and Zhizhong Su and Wenyu Liu and Xinggang Wang},
  journal={arXiv preprint arXiv:2602.21186},
  year=2026
}
```


## License

This project is released under the [MIT](LICENSE) License.
