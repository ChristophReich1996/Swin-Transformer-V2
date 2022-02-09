from typing import Tuple, List

import torch
import torch.nn as nn

from swin_transformer_v2.model_parts import PatchEmbedding, SwinTransformerStage

__all__: List[str] = ["SwinTransformerV2"]


class SwinTransformerV2(nn.Module):
    """
    This class implements the Swin Transformer without classification head.
    """

    def __init__(self,
                 in_channels: int,
                 embedding_channels: int,
                 depths: Tuple[int, ...],
                 input_resolution: Tuple[int, int],
                 number_of_heads: Tuple[int, ...],
                 window_size: int = 7,
                 patch_size: int = 4,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.2,
                 use_checkpoint: bool = False,
                 sequential_self_attention: bool = False,
                 use_deformable_block: bool = False) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param depth: (int) Depth of the stage (number of layers)
        :param downscale: (bool) If true input is downsampled (see Fig. 3 or V1 paper)
        :param input_resolution: (Tuple[int, int]) Input resolution
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param use_checkpoint: (bool) If true checkpointing is utilized
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        :param use_deformable_block: (bool) If true deformable block is used
        """
        # Call super constructor
        super(SwinTransformerV2, self).__init__()
        # Save parameters
        self.patch_size: int = patch_size
        # Init patch embedding
        self.patch_embedding: nn.Module = PatchEmbedding(in_channels=in_channels, out_channels=embedding_channels,
                                                         patch_size=patch_size)
        # Compute patch resolution
        patch_resolution: Tuple[int, int] = (input_resolution[0] // patch_size, input_resolution[1] // patch_size)
        # Path dropout dependent on depth
        dropout_path = torch.linspace(0., dropout_path, sum(depths)).tolist()
        # Init stages
        self.stages: nn.ModuleList = nn.ModuleList()
        for index, (depth, number_of_head) in enumerate(zip(depths, number_of_heads)):
            self.stages.append(
                SwinTransformerStage(
                    in_channels=embedding_channels * (2 ** max(index - 1, 0)),
                    depth=depth,
                    downscale=not (index == 0),
                    input_resolution=(patch_resolution[0] // (2 ** max(index - 1, 0)),
                                      patch_resolution[1] // (2 ** max(index - 1, 0))),
                    number_of_heads=number_of_head,
                    window_size=window_size,
                    ff_feature_ratio=ff_feature_ratio,
                    dropout=dropout,
                    dropout_attention=dropout_attention,
                    dropout_path=dropout_path[sum(depths[:index]):sum(depths[:index + 1])],
                    use_checkpoint=use_checkpoint,
                    sequential_self_attention=sequential_self_attention,
                    use_deformable_block=use_deformable_block and (index > 0)
                ))

    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Compute new patch resolution
        new_patch_resolution: Tuple[int, int] = (new_input_resolution[0] // self.patch_size,
                                                 new_input_resolution[1] // self.patch_size)
        # Update resolution of each stage
        for index, stage in enumerate(self.stages):  # type: int, SwinTransformerStage
            stage.update_resolution(new_window_size=new_window_size,
                                    new_input_resolution=(new_patch_resolution[0] // (2 ** max(index - 1, 0)),
                                                          new_patch_resolution[1] // (2 ** max(index - 1, 0))))

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :return: (List[torch.Tensor]) List of features from each stage
        """
        # Perform patch embedding
        output: torch.Tensor = self.patch_embedding(input)
        # Init list to store feature
        features: List[torch.Tensor] = []
        # Forward pass of each stage
        for stage in self.stages:
            output: torch.Tensor = stage(output)
            features.append(output)
        return features


def swin_transformer_v2_t(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a tiny Swin Transformer V2 (SwinV2-T: C = 96, layer numbers = {2, 2, 6, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Tiny Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=96,
                             depths=(2, 2, 6, 2),
                             number_of_heads=(3, 6, 12, 24),
                             **kwargs)


def swin_transformer_v2_s(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a small Swin Transformer V2 (SwinV2-S: C = 96, layer numbers ={2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Small Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=96,
                             depths=(2, 2, 18, 2),
                             number_of_heads=(3, 6, 12, 24),
                             **kwargs)


def swin_transformer_v2_b(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a base Swin Transformer V2 (SwinV2-B: C = 128, layer numbers ={2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Base Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=128,
                             depths=(2, 2, 18, 2),
                             number_of_heads=(4, 8, 16, 32),
                             **kwargs)


def swin_transformer_v2_l(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a large Swin Transformer V2 (SwinV2-L: C = 192, layer numbers ={2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Large Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=192,
                             depths=(2, 2, 18, 2),
                             number_of_heads=(6, 12, 24, 48),
                             **kwargs)


def swin_transformer_v2_h(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a large Swin Transformer V2 (SwinV2-H: C = 352, layer numbers = {2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Large Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=352,
                             depths=(2, 2, 18, 2),
                             number_of_heads=(11, 22, 44, 88),
                             **kwargs)


def swin_transformer_v2_g(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a giant Swin Transformer V2 (SwinV2-G: C = 512, layer numbers = {2, 2, 42, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Giant Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=512,
                             depths=(2, 2, 42, 2),
                             number_of_heads=(16, 32, 64, 128),
                             **kwargs)
