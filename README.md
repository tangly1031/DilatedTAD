# DilatedTAD: Enhancing Adaptability to Actions of Varying Durations for Temporal Action Detection

DilatedTAD is a temporal action detection framework that improves long- and short-duration action localization by using multi-branch dilated Mamba modules and bidirectional modeling. 

## 🛠️ Installation

Please refer to [install.md](docs/en/install.md) for installation and data preparation.


## 🚀 Usage

Please refer to [usage.md](docs/en/usage.md) for details of training and evaluation scripts.


## 📂 Dataset Preparation

Configure the dataset for the **THUMOS** environment: [OpenTAD Dataset Setup Guide](https://github.com/sming256/OpenTAD)


## 🚀 Training Command (THUMOS Dataset)

```bashdtad
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/dtad/thumos_internvideo6b.py
```

## 🖊️ Citation


If you think this repo is helpful, please cite us:

```bibtex
@ARTICLE{11113253,
  author={Tang, Longyang and Zhang, Bo and Lv, Hui and Xu, Rui and Tian, Xudong and Zhou, Junsheng and Chen, Yi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={DilatedTAD: Enhancing Adaptability to Actions of Varying Durations for Temporal Action Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Computational modeling;Videos;Redundancy;Transformers;Sensitivity;Convolution;Computational efficiency;Spatiotemporal phenomena;Proposals;Feature extraction;Temporal action detection;video understanding;mamba},
  doi={10.1109/TCSVT.2025.3595931}}
```

If you have any questions, please contact: `tangly@nnu.edu.cn`.

