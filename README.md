# DilatedTAD: Enhancing Adaptability to Actions of Varying Durations for Temporal Action Detection

DilatedTAD is a temporal action detection framework that improves long- and short-duration action localization by using multi-branch dilated Mamba modules and bidirectional modeling. It effectively balances temporal context and efficiency, outperforming state-of-the-art methods on multiple benchmarks.

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

## 🖊️ Citation

If you think this repo is helpful, please cite us:

```bibtex
@misc{2024opentad,
    title={OpenTAD: An Open-Source Toolbox for Temporal Action Detection},
    author={Shuming Liu, Chen Zhao, Fatimah Zohra, Mattia Soldan, Carlos Hinojosa, Alejandro Pardo, Anthony Cioppa, Lama Alssum, Mengmeng Xu, Merey Ramazanova, Juan León Alcázar, Silvio Giancola, Bernard Ghanem},
    howpublished = {\url{https://github.com/sming256/opentad}},
    year={2024}
}
```

If you have any questions, please contact: `tangly@nnu.edu.cn`.
