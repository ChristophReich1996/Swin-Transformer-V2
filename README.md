# Swin Transformer V2: Scaling Up Capacity and Resolution

Unofficial **PyTorch** reimplementation of the
paper [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883.pdf)
by Ze Liu, Han Hu et al. (Microsoft Research Asia).

**This repository includes a pure PyTorch implementation of the Swin Transformer V2.**

The official Swin Transformer V1 implementation is available [here](https://github.com/microsoft/Swin-Transformer).
Currently (10.01.2022), an official implementation of the Swin Transformer V2 is not publicly available.

## Installation

You can simply install the Swin Transformer V2 implementation as a Python package by using `pip`.

```shell script
pip install git+https://github.com/ChristophReich1996/Involution
```

Alternatively, you can clone the repository and use the implementation in [swin_transformer_v2](swin_transformer_v2) directly in your project.

## Usage

This implementation provides the configurations reported in the paper (SwinV2-T, SwinV2-S, etc.). You can build the
model by calling the corresponding function. Please note that the Swin Transformer V2 (`SwinTransformerV2` class) 
implementation returns the feature maps of each stage of the network (`List[torch.Tensor]`). If you want to use this 
implementation for image classification simply wrap this model and take the final feature map.

```python
from swin_transformer_v2 import SwinTransformerV2

from swin_transformer_v2 import swin_transformer_v2_t, swin_transformer_v2_s, swin_transformer_v2_b, \
    swin_transformer_v2_l, swin_transformer_v2_h, swin_transformer_v2_g

# SwinV2-T
swin_transformer: SwinTransformerV2 = swin_transformer_v2_t(in_channels=3,
                                                            window_size=8,
                                                            input_resolution=(256, 256),
                                                            sequential_self_attention=False,
                                                            use_checkpoint=False)
```

If you want to change the resolution and/or the window size for fine-tuning or inference pleas use the `update_resolution` method.

```python
# Change resolution and window size of the model
swin_transformer.update_resolution(new_window_size=16, new_input_resolution=(512, 512))
```

In case you want to use a custom configuration you can use the `SwinTransformerV2` class. The constructor method takes 
the following parameters.

| Parameter | Description | Type |
| ------------- | ------------- | ------------- |
| in_channels | Number of input channels | int |
| depth | Depth of the stage (number of layers) | int |
| downscale | If true input is downsampled (see Fig. 3 or V1 paper) | bool |
| input_resolution | Input resolution | Tuple[int, int] |
| number_of_heads | Number of attention heads to be utilized | int |
| window_size | Window size to be utilized | int |
| shift_size | Shifting size to be used | int |
| ff_feature_ratio | Ratio of the hidden dimension in the FFN to the input channels | int |
| dropout | Dropout in input mapping | float |
| dropout_attention | Dropout rate of attention map | float |
| dropout_path | Dropout in main path | float |
| use_checkpoint | If true checkpointing is utilized | bool |
| sequential_self_attention | If true sequential self-attention is performed | bool |

[This file](example.py) includes a full example how to use this implementation.

## Disclaimer

This is a very experimental implementation based on the [Swin Transformer V2 paper](https://arxiv.org/pdf/2111.09883.pdf) and the [official implementation of the Swin Transformer V1](https://github.com/microsoft/Swin-Transformer).
Since an official implementation of the Swin Transformer V2 is not yet published, it is not possible to say to which extent this implementation might differ from the original one. If you have any issues with this implementation please raise an issue.

## Reference

```bibtex
@article{Liu2021,
    title={{Swin Transformer V2: Scaling Up Capacity and Resolution}},
    author={Liu, Ze and Hu, Han and Lin, Yutong and Yao, Zhuliang and Xie, Zhenda and Wei, Yixuan and Ning, Jia and Cao, 
            Yue and Zhang, Zheng and Dong, Li and others},
    journal={arXiv preprint arXiv:2111.09883},
    year={2021}
}
```