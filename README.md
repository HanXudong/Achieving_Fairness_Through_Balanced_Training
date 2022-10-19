# Balancing out Bias: Achieving_Fairness_Through_Balanced_Training

Source codes for EMNLP 2022 paper "Balancing out Bias: Achieving Fairness Through Balanced Training"

If you use the code, please cite the following paper:

```
@article{han2021balancing,
  title={Balancing out Bias: Achieving Fairness Through Training Reweighting},
  author={Xudong Han and Timothy Baldwin and Trevor Cohn},
  journal={ArXiv},
  year={2021},
  volume={arXiv preprint arXiv:2109.08253}
}
```

# Quick Links
+ [Overview](#overview)

+ [Requirements](#requirements)

+ [Data Preparation](#data-preparation)

+ [Source Code](#source-code)

+ [Experiments](#experiments)

# Overview

In this work, we first propose a framework to balance demographic groups within each target class in training, and then present a demographic-augmented model, which further improves fairness.

# Requirements

The model is implemented using PyTorch and FairLib.

```
tqdm==4.62.3
numpy==1.22
docopt==0.6.2
pandas==1.3.4
scikit-learn==1.0
torch==1.10.0
PyYAML==6.0
seaborn==0.11.2
matplotlib==3.5.0
pickle5==0.0.12
transformers==4.11.3
sacremoses==0.0.53
```

Alternatively, you can install the fairlib directly:
```
pip install fairlib
```

# Data Preparation

```python
from fairlib import datasets

datasets.prepare_dataset("moji", "data/deepmoji")
datasets.prepare_dataset("bios", "data/bios")

```

# Source Code

## Balanced training

For a particular balanced training objective, e.g., joint balance, both adjusting instances weights and sampling can be employed. 

Our implementations are presented in `src\balanced_training.py` , where the `get_weights` function and `get_sampled_indices` correspond to instances reweighting and resampling respectively.
Specifically,
- `get_weights`: Given the balanced training objective, target labels, and protected labels, pre-calculate weights for each instance.
- `get_sample_indices`: Given the balanced training objective, target labels, and protected labels, sampling instances for each group.

The implemented objectives includes:
- **y**: balance the target label distribution, which is typically used for long-tail learning
- **g**: balance the protected label distributions
- **joint** jointly balance protected attributes and target labels
- **stratified_y** balance protected labels within each target class
- **stratified_g** balance target labels within in each protected group
- **EO** maintain as much instances as possible for downsampling but degrade to *stratified_y* for reweighting and resampling.

## Demographic-augmentation

The gated model learns a shared encoder and one extra encoder for each demographic groups. 
Our implementation is presented in `augmentation_layer.py`, which augment the naive encoder with demographic-specific encoders.

When using the augmentation layer, it takes the same inputs as the shared encoder along with groups labels, its outputs then will be added to the outputs of the main model to augment demographic information.
```python
        # Augmentation
        if self.args.gated and self.args.n_hidden > 0:
            assert group_label is not None, "Group labels are needed for augmentation"

            specific_output = self.augmentation_components(input_data, group_label)

            main_output = main_output + specific_output
```

# Experiments

All experiments can be reproduced by using fairlib.

- [Balanced training](https://hanxudong.github.io/fairlib/tutorial_usage.html#train-a-model-with-balanced-training)
    ```bash
    python fairlib --BT Reweighting --BTObj joint
    ```

- [Gated Model](https://hanxudong.github.io/fairlib/tutorial_usage.html#train-a-model-to-incorporate-demographic-factors)
    ```
    python fairlib --BT --BTObj joint --gated
    ```
