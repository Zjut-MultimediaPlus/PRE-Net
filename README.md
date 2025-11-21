# PRE-Net

## Overview

This repository contains the code for the research presented in the paper **[From Swath to Full-Disc: Advancing Precipitation Retrieval with Multimodal Knowledge Expansion]**. This task leverages the complementary strengths of multimodal data (IR, PMW, and PR) to address the constraints of narrow coverage and enhance retrieval accuracy. It includes the code for training and testing the model, as well as the necessary scripts and configurations for easy execution.

## Dataset & Model Weights

The dataset and pre-trained model weights for this project are not included in the repository due to their large size. You can download them from the following links:

* [Dataset Link (Baidu Cloud)](YOUR_BAIDU_CLOUD_LINK)
* [Model Weights Link (Baidu Cloud)](YOUR_BAIDU_CLOUD_LINK)

Please follow the instructions in the links to download and place the files in the appropriate directory (e.g., `data/` for dataset, `checkpoints/` for model weights).

## Usage

### Training

To train the model, run the `train.sh` script. This will automatically start the training process with the available dataset.

```bash
bash train.sh
```

### Testing

After training, you can use the `test.sh` script to evaluate the model performance on the test dataset.

```bash
bash test.sh
```

## Citation

If you use this code in your research, please cite the following paper:

```
@article{wang2025swath,
  title={From swath to full-disc: Advancing precipitation retrieval with multimodal knowledge expansion},
  author={Wang, Zheng and Ying, Kai and Xu, Bin and Wang, Chunjiao and Bai, Cong},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={3091--3101},
  year={2025}
}
```
