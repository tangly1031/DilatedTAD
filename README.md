# DilatedTAD: Enhancing Adaptability to Actions of Varying Durations for Temporal Action Detection

DilatedTAD is a temporal action detection framework that improves long- and short-duration action localization by using multi-branch dilated Mamba modules and bidirectional modeling. 

## 🛠️ Installation

Please refer to [install.md](docs/en/install.md) for installation and data preparation.


## 🚀 Usage

Please refer to [usage.md](docs/en/usage.md) for details of training and evaluation scripts.


## 📂 Dataset Preparation

Configure the dataset for the **THUMOS** environment: [OpenTAD Dataset Setup Guide](https://github.com/sming256/OpenTAD)


## 🚀 Training Command (THUMOS Dataset)

```bash
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/DilatedTAD/thumos_internvideo6b.py
```

If you have any questions, please contact: `tangly@nnu.edu.cn`.
