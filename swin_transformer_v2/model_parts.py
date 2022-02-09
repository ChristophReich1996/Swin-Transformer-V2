from typing import Tuple, Optional, List, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm

__all__: List[str] = ["SwinTransformerStage", "SwinTransformerBlock", "DeformableSwinTransformerBlock"]


class FeedForward(nn.Sequential):
    """
    Feed forward module used in the transformer encoder.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float = 0.) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param hidden_features: (int) Number of hidden features
        :param out_features: (int) Number of output features
        :param dropout: (float) Dropout factor
        """
        # Call super constructor and init modules
        super().__init__(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Dropout(p=dropout)
        )


def bchw_to_bhwc(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, height, width, channels]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, height, width, channels]
    """
    return input.permute(0, 2, 3, 1)


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2)


def unfold(input: torch.Tensor,
           window_size: int) -> torch.Tensor:
    """
    Unfolds (non-overlapping) a given feature map by the given window size (stride = window size)
    :param input: (torch.Tensor) Input feature map of the shape [batch size, channels, height, width]
    :param window_size: (int) Window size to be applied
    :return: (torch.Tensor) Unfolded tensor of the shape [batch size * windows, channels, window size, window size]
    """
    # Get original shape
    _, channels, height, width = input.shape  # type: int, int, int, int
    # Unfold input
    output: torch.Tensor = input.unfold(dimension=3, size=window_size, step=window_size) \
        .unfold(dimension=2, size=window_size, step=window_size)
    # Reshape to [batch size * windows, channels, window size, window size]
    output: torch.Tensor = output.permute(0, 2, 3, 1, 5, 4).reshape(-1, channels, window_size, window_size)
    return output


def fold(input: torch.Tensor,
         window_size: int,
         height: int,
         width: int) -> torch.Tensor:
    """
    Fold a tensor of windows again to a 4D feature map
    :param input: (torch.Tensor) Input tensor of windows [batch size * windows, channels, window size, window size]
    :param window_size: (int) Window size to be reversed
    :param height: (int) Height of the feature map
    :param width: (int) Width of the feature map
    :return: (torch.Tensor) Folded output tensor of the shape [batch size, channels, height, width]
    """
    # Get channels of windows
    channels: int = input.shape[1]
    # Get original batch size
    batch_size: int = int(input.shape[0] // (height * width // window_size // window_size))
    # Reshape input to
    output: torch.Tensor = input.view(batch_size, height // window_size, width // window_size, channels,
                                      window_size, window_size)
    output: torch.Tensor = output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
    return output


class WindowMultiHeadAttention(nn.Module):
    """
    This class implements window-based Multi-Head-Attention.
    """

    def __init__(self,
                 in_features: int,
                 window_size: int,
                 number_of_heads: int,
                 dropout_attention: float = 0.,
                 dropout_projection: float = 0.,
                 meta_network_hidden_features: int = 256,
                 sequential_self_attention: bool = False) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param window_size: (int) Window size
        :param number_of_heads: (int) Number of attention heads
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_projection: (float) Dropout rate after projection
        :param meta_network_hidden_features: (int) Number of hidden features in the two layer MLP meta network
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        """
        # Call super constructor
        super(WindowMultiHeadAttention, self).__init__()
        # Check parameter
        assert (in_features % number_of_heads) == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (number_of_heads)."
        # Save parameters
        self.in_features: int = in_features
        self.window_size: int = window_size
        self.number_of_heads: int = number_of_heads
        self.sequential_self_attention: bool = sequential_self_attention
        # Init query, key and value mapping as a single layer
        self.mapping_qkv: nn.Module = nn.Linear(in_features=in_features, out_features=in_features * 3, bias=True)
        # Init attention dropout
        self.attention_dropout: nn.Module = nn.Dropout(dropout_attention)
        # Init projection mapping
        self.projection: nn.Module = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        # Init projection dropout
        self.projection_dropout: nn.Module = nn.Dropout(dropout_projection)
        # Init meta network for positional encodings
        self.meta_network: nn.Module = nn.Sequential(
            nn.Linear(in_features=2, out_features=meta_network_hidden_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=meta_network_hidden_features, out_features=number_of_heads, bias=True))
        # Init tau
        self.register_parameter("tau", torch.nn.Parameter(torch.ones(1, number_of_heads, 1, 1)))
        # Init pair-wise relative positions (log-spaced)
        self.__make_pair_wise_relative_positions()

    def __make_pair_wise_relative_positions(self) -> None:
        """
        Method initializes the pair-wise relative positions to compute the positional biases
        """
        indexes: torch.Tensor = torch.arange(self.window_size, device=self.tau.device)
        coordinates: torch.Tensor = torch.stack(torch.meshgrid([indexes, indexes]), dim=0)
        coordinates: torch.Tensor = torch.flatten(coordinates, start_dim=1)
        relative_coordinates: torch.Tensor = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates: torch.Tensor = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log: torch.Tensor = torch.sign(relative_coordinates) \
                                                 * torch.log(1. + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log)

    def update_resolution(self,
                          new_window_size: int,
                          **kwargs: Any) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param kwargs: (Any) Unused
        """
        # Set new window size
        self.window_size: int = new_window_size
        # Make new pair-wise relative positions
        self.__make_pair_wise_relative_positions()

    def __get_relative_positional_encodings(self) -> torch.Tensor:
        """
        Method computes the relative positional encodings
        :return: (torch.Tensor) Relative positional encodings [1, number of heads, window size ** 2, window size ** 2]
        """
        relative_position_bias: torch.Tensor = self.meta_network(self.relative_coordinates_log)
        relative_position_bias: torch.Tensor = relative_position_bias.permute(1, 0)
        relative_position_bias: torch.Tensor = relative_position_bias.reshape(self.number_of_heads,
                                                                              self.window_size * self.window_size,
                                                                              self.window_size * self.window_size)
        return relative_position_bias.unsqueeze(0)

    def __self_attention(self,
                         query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         batch_size_windows: int,
                         tokens: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function performs standard (non-sequential) scaled cosine self-attention
        :param query: (torch.Tensor) Query tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param key: (torch.Tensor) Key tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param value: (torch.Tensor) Value tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param batch_size_windows: (int) Size of the first dimension of the input tensor (batch size * windows)
        :param tokens: (int) Number of tokens in the input
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output feature map of the shape [batch size * windows, tokens, channels]
        """
        # Compute attention map with scaled cosine attention
        attention_map: torch.Tensor = torch.einsum("bhqd, bhkd -> bhqk", query, key) \
                                      / torch.maximum(torch.norm(query, dim=-1, keepdim=True)
                                                      * torch.norm(key, dim=-1, keepdim=True).transpose(-2, -1),
                                                      torch.tensor(1e-06, device=query.device, dtype=query.dtype))
        attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)
        # Apply relative positional encodings
        attention_map: torch.Tensor = attention_map + self.__get_relative_positional_encodings()
        # Apply mask if utilized
        if mask is not None:
            number_of_windows: int = mask.shape[0]
            attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows, number_of_windows,
                                                             self.number_of_heads, tokens, tokens)
            attention_map: torch.Tensor = attention_map + mask.unsqueeze(1).unsqueeze(0)
            attention_map: torch.Tensor = attention_map.view(-1, self.number_of_heads, tokens, tokens)
        attention_map: torch.Tensor = attention_map.softmax(dim=-1)
        # Perform attention dropout
        attention_map: torch.Tensor = self.attention_dropout(attention_map)
        # Apply attention map and reshape
        output: torch.Tensor = torch.einsum("bhal, bhlv -> bhav", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def __sequential_self_attention(self,
                                    query: torch.Tensor,
                                    key: torch.Tensor,
                                    value: torch.Tensor,
                                    batch_size_windows: int,
                                    tokens: int,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function performs sequential scaled cosine self-attention
        :param query: (torch.Tensor) Query tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param key: (torch.Tensor) Key tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param value: (torch.Tensor) Value tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param batch_size_windows: (int) Size of the first dimension of the input tensor (batch size * windows)
        :param tokens: (int) Number of tokens in the input
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output feature map of the shape [batch size * windows, tokens, channels]
        """
        # Init output tensor
        output: torch.Tensor = torch.ones_like(query)
        # Compute relative positional encodings fist
        relative_position_bias: torch.Tensor = self.__get_relative_positional_encodings()
        # Iterate over query and key tokens
        for token_index_query in range(tokens):
            # Compute attention map with scaled cosine attention
            attention_map: torch.Tensor = \
                torch.einsum("bhd, bhkd -> bhk", query[:, :, token_index_query], key) \
                / torch.maximum(torch.norm(query[:, :, token_index_query], dim=-1, keepdim=True)
                                * torch.norm(key, dim=-1, keepdim=False),
                                torch.tensor(1e-06, device=query.device, dtype=query.dtype))
            attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)[..., 0]
            # Apply positional encodings
            attention_map: torch.Tensor = attention_map + relative_position_bias[..., token_index_query, :]
            # Apply mask if utilized
            if mask is not None:
                number_of_windows: int = mask.shape[0]
                attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows,
                                                                 number_of_windows, self.number_of_heads, 1,
                                                                 tokens)
                attention_map: torch.Tensor = attention_map \
                                              + mask.unsqueeze(1).unsqueeze(0)[..., token_index_query, :].unsqueeze(3)
                attention_map: torch.Tensor = attention_map.view(-1, self.number_of_heads, tokens)
            attention_map: torch.Tensor = attention_map.softmax(dim=-1)
            # Perform attention dropout
            attention_map: torch.Tensor = self.attention_dropout(attention_map)
            # Apply attention map and reshape
            output[:, :, token_index_query] = torch.einsum("bhl, bhlv -> bhv", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size * windows, channels, height, width]
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output tensor of the shape [batch size * windows, channels, height, width]
        """
        # Save original shape
        batch_size_windows, channels, height, width = input.shape  # type: int, int, int, int
        tokens: int = height * width
        # Reshape input to [batch size * windows, tokens (height * width), channels]
        input: torch.Tensor = input.reshape(batch_size_windows, channels, tokens).permute(0, 2, 1)
        # Perform query, key, and value mapping
        query_key_value: torch.Tensor = self.mapping_qkv(input)
        query_key_value: torch.Tensor = query_key_value.view(batch_size_windows, tokens, 3, self.number_of_heads,
                                                             channels // self.number_of_heads).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]
        # Perform attention
        if self.sequential_self_attention:
            output: torch.Tensor = self.__sequential_self_attention(query=query, key=key, value=value,
                                                                    batch_size_windows=batch_size_windows,
                                                                    tokens=tokens,
                                                                    mask=mask)
        else:
            output: torch.Tensor = self.__self_attention(query=query, key=key, value=value,
                                                         batch_size_windows=batch_size_windows, tokens=tokens,
                                                         mask=mask)
        # Perform linear mapping and dropout
        output: torch.Tensor = self.projection_dropout(self.projection(output))
        # Reshape output to original shape [batch size * windows, channels, height, width]
        output: torch.Tensor = output.permute(0, 2, 1).view(batch_size_windows, channels, height, width)
        return output


class SwinTransformerBlock(nn.Module):
    """
    This class implements the Swin transformer block.
    """

    def __init__(self,
                 in_channels: int,
                 input_resolution: Tuple[int, int],
                 number_of_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.0,
                 sequential_self_attention: bool = False) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param input_resolution: (Tuple[int, int]) Input resolution
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        """
        # Call super constructor
        super(SwinTransformerBlock, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.input_resolution: Tuple[int, int] = input_resolution
        # Catch case if resolution is smaller than the window size
        if min(self.input_resolution) <= window_size:
            self.window_size: int = min(self.input_resolution)
            self.shift_size: int = 0
            self.make_windows: bool = False
        else:
            self.window_size: int = window_size
            self.shift_size: int = shift_size
            self.make_windows: bool = True
        # Init normalization layers
        self.normalization_1: nn.Module = nn.LayerNorm(normalized_shape=in_channels)
        self.normalization_2: nn.Module = nn.LayerNorm(normalized_shape=in_channels)
        # Init window attention module
        self.window_attention: WindowMultiHeadAttention = WindowMultiHeadAttention(
            in_features=in_channels,
            window_size=self.window_size,
            number_of_heads=number_of_heads,
            dropout_attention=dropout_attention,
            dropout_projection=dropout,
            sequential_self_attention=sequential_self_attention)
        # Init dropout layer
        self.dropout: nn.Module = timm.models.layers.DropPath(
            drop_prob=dropout_path) if dropout_path > 0. else nn.Identity()
        # Init feed-forward network
        self.feed_forward_network: nn.Module = FeedForward(in_features=in_channels,
                                                           hidden_features=int(in_channels * ff_feature_ratio),
                                                           dropout=dropout,
                                                           out_features=in_channels)
        # Make attention mask
        self.__make_attention_mask()

    def __make_attention_mask(self) -> None:
        """
        Method generates the attention mask used in shift case
        """
        # Make masks for shift case
        if self.shift_size > 0:
            height, width = self.input_resolution  # type: int, int
            mask: torch.Tensor = torch.zeros(height, width, device=self.window_attention.tau.device)
            height_slices: Tuple = (slice(0, -self.window_size),
                                    slice(-self.window_size, -self.shift_size),
                                    slice(-self.shift_size, None))
            width_slices: Tuple = (slice(0, -self.window_size),
                                   slice(-self.window_size, -self.shift_size),
                                   slice(-self.shift_size, None))
            counter: int = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    mask[height_slice, width_slice] = counter
                    counter += 1
            mask_windows: torch.Tensor = unfold(mask[None, None], self.window_size)
            mask_windows: torch.Tensor = mask_windows.reshape(-1, self.window_size * self.window_size)
            attention_mask: Optional[torch.Tensor] = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask == 0, float(0.0))
        else:
            attention_mask: Optional[torch.Tensor] = None
        # Save mask
        self.register_buffer("attention_mask", attention_mask)

    def update_resolution(self,
                          new_window_size: int,
                          new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update input resolution
        self.input_resolution: Tuple[int, int] = new_input_resolution
        # Catch case if resolution is smaller than the window size
        if min(self.input_resolution) <= new_window_size:
            self.window_size: int = min(self.input_resolution)
            self.shift_size: int = 0
            self.make_windows: bool = False
        else:
            self.window_size: int = new_window_size
            self.shift_size: int = self.shift_size
            self.make_windows: bool = True
        # Update attention mask
        self.__make_attention_mask()
        # Update attention module
        self.window_attention.update_resolution(new_window_size=new_window_size)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, in channels, height, width]
        """
        # Save shape
        batch_size, channels, height, width = input.shape  # type: int, int, int, int
        # Shift input if utilized
        if self.shift_size > 0:
            output_shift: torch.Tensor = torch.roll(input=input, shifts=(-self.shift_size, -self.shift_size),
                                                    dims=(-1, -2))
        else:
            output_shift: torch.Tensor = input
        # Make patches
        output_patches: torch.Tensor = unfold(input=output_shift, window_size=self.window_size) \
            if self.make_windows else output_shift
        # Perform window attention
        output_attention: torch.Tensor = self.window_attention(output_patches, mask=self.attention_mask)
        # Merge patches
        output_merge: torch.Tensor = fold(input=output_attention, window_size=self.window_size, height=height,
                                          width=width) if self.make_windows else output_attention
        # Reverse shift if utilized
        if self.shift_size > 0:
            output_shift: torch.Tensor = torch.roll(input=output_merge, shifts=(self.shift_size, self.shift_size),
                                                    dims=(-1, -2))
        else:
            output_shift: torch.Tensor = output_merge
        # Perform normalization
        output_normalize: torch.Tensor = self.normalization_1(output_shift.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Skip connection
        output_skip: torch.Tensor = self.dropout(output_normalize) + input
        # Feed forward network, normalization and skip connection
        output_feed_forward: torch.Tensor = self.feed_forward_network(
            output_skip.view(batch_size, channels, -1).permute(0, 2, 1)).permute(0, 2, 1)
        output_feed_forward: torch.Tensor = output_feed_forward.view(batch_size, channels, height, width)
        output_normalize: torch.Tensor = bhwc_to_bchw(self.normalization_2(bchw_to_bhwc(output_feed_forward)))
        output: torch.Tensor = output_skip + self.dropout(output_normalize)
        return output


class DeformableSwinTransformerBlock(SwinTransformerBlock):
    """
    This class implements a deformable version of the Swin Transformer block.
    Inspired by: https://arxiv.org/pdf/2201.00520.pdf
    """

    def __init__(self,
                 in_channels: int,
                 input_resolution: Tuple[int, int],
                 number_of_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.0,
                 sequential_self_attention: bool = False,
                 offset_downscale_factor: int = 2) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param input_resolution: (Tuple[int, int]) Input resolution
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        :param offset_downscale_factor: (int) Downscale factor of offset network
        """
        # Call super constructor
        super(DeformableSwinTransformerBlock, self).__init__(
            in_channels=in_channels,
            input_resolution=input_resolution,
            number_of_heads=number_of_heads,
            window_size=window_size,
            shift_size=shift_size,
            ff_feature_ratio=ff_feature_ratio,
            dropout=dropout,
            dropout_attention=dropout_attention,
            dropout_path=dropout_path,
            sequential_self_attention=sequential_self_attention
        )
        # Save parameter
        self.offset_downscale_factor: int = offset_downscale_factor
        self.number_of_heads: int = number_of_heads
        # Make default offsets
        self.__make_default_offsets()
        # Init offset network
        self.offset_network: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=offset_downscale_factor,
                      padding=3, groups=in_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=2 * self.number_of_heads, kernel_size=1, stride=1,
                      padding=0, bias=True)
        )

    def __make_default_offsets(self) -> None:
        """
        Method generates the default sampling grid (inspired by kornia)
        """
        # Init x and y coordinates
        x: torch.Tensor = torch.linspace(0, self.input_resolution[1] - 1, self.input_resolution[1],
                                         device=self.window_attention.tau.device)
        y: torch.Tensor = torch.linspace(0, self.input_resolution[0] - 1, self.input_resolution[0],
                                         device=self.window_attention.tau.device)
        # Normalize coordinates to a range of [-1, 1]
        x: torch.Tensor = (x / (self.input_resolution[1] - 1) - 0.5) * 2
        y: torch.Tensor = (y / (self.input_resolution[0] - 1) - 0.5) * 2
        # Make grid [2, height, width]
        grid: torch.Tensor = torch.stack(torch.meshgrid([x, y])).transpose(1, 2)
        # Reshape grid to [1, height, width, 2]
        grid: torch.Tensor = grid.unsqueeze(dim=0).permute(0, 2, 3, 1)
        # Register in module
        self.register_buffer("default_grid", grid)

    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update resolution and window size
        super(DeformableSwinTransformerBlock, self).update_resolution(new_window_size=new_window_size,
                                                                      new_input_resolution=new_input_resolution)
        # Update default sampling grid
        self.__make_default_offsets()

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        # Get input shape
        batch_size, channels, height, width = input.shape
        # Compute offsets of the shape [batch size, 2, height / r, width / r]
        offsets: torch.Tensor = self.offset_network(input)
        # Upscale offsets to the shape [batch size, 2 * number of heads, height, width]
        offsets: torch.Tensor = F.interpolate(input=offsets,
                                              size=(height, width), mode="bilinear", align_corners=True)
        # Reshape offsets to [batch size, number of heads, height, width, 2]
        offsets: torch.Tensor = offsets.reshape(batch_size, -1, 2, height, width).permute(0, 1, 3, 4, 2)
        # Flatten batch size and number of heads and apply tanh
        offsets: torch.Tensor = offsets.view(-1, height, width, 2).tanh()
        # Cast offset grid to input data type
        if input.dtype != self.default_grid.dtype:
            self.default_grid = self.default_grid.type(input.dtype)
        # Construct offset grid
        offset_grid: torch.Tensor = self.default_grid.repeat_interleave(repeats=offsets.shape[0], dim=0) + offsets
        # Reshape input to [batch size * number of heads, channels / number of heads, height, width]
        input: torch.Tensor = input.view(batch_size, self.number_of_heads, channels // self.number_of_heads, height,
                                         width).flatten(start_dim=0, end_dim=1)
        # Apply sampling grid
        input_resampled: torch.Tensor = F.grid_sample(input=input, grid=offset_grid.clip(min=-1, max=1),
                                                      mode="bilinear", align_corners=True, padding_mode="reflection")
        # Reshape resampled tensor again to [batch size, channels, height, width]
        input_resampled: torch.Tensor = input_resampled.view(batch_size, channels, height, width)
        return super(DeformableSwinTransformerBlock, self).forward(input=input_resampled)


class PatchMerging(nn.Module):
    """
    This class implements the patch merging approach which is essential a strided convolution with normalization before
    """

    def __init__(self,
                 in_channels: int) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        """
        # Call super constructor
        super(PatchMerging, self).__init__()
        # Init normalization
        self.normalization: nn.Module = nn.LayerNorm(normalized_shape=4 * in_channels)
        # Init linear mapping
        self.linear_mapping: nn.Module = nn.Linear(in_features=4 * in_channels, out_features=2 * in_channels,
                                                   bias=False)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, 2 * in channels, height // 2, width // 2]
        """
        # Get original shape
        batch_size, channels, height, width = input.shape  # type: int, int, int, int
        # Reshape input to [batch size, in channels, height, width]
        input: torch.Tensor = bchw_to_bhwc(input)
        # Unfold input
        input: torch.Tensor = input.unfold(dimension=1, size=2, step=2).unfold(dimension=2, size=2, step=2)
        input: torch.Tensor = input.reshape(batch_size, input.shape[1], input.shape[2], -1)
        # Normalize input
        input: torch.Tensor = self.normalization(input)
        # Perform linear mapping
        output: torch.Tensor = bhwc_to_bchw(self.linear_mapping(input))
        return output


class PatchEmbedding(nn.Module):
    """
    Module embeds a given image into patch embeddings.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 96,
                 patch_size: int = 4) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param patch_size: (int) Patch size to be utilized
        :param image_size: (int) Image size to be used
        """
        # Call super constructor
        super(PatchEmbedding, self).__init__()
        # Save parameters
        self.out_channels: int = out_channels
        # Init linear embedding as a convolution
        self.linear_embedding: nn.Module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=(patch_size, patch_size),
                                                     stride=(patch_size, patch_size))
        # Init layer normalization
        self.normalization: nn.Module = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass transforms a given batch of images into a patch embedding
        :param input: (torch.Tensor) Input images of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Patch embedding of the shape [batch size, patches + 1, out channels]
        """
        # Perform linear embedding
        embedding: torch.Tensor = self.linear_embedding(input)
        # Perform normalization
        embedding: torch.Tensor = bhwc_to_bchw(self.normalization(bchw_to_bhwc(embedding)))
        return embedding


class SwinTransformerStage(nn.Module):
    """
    This class implements a stage of the Swin transformer including multiple layers.
    """

    def __init__(self,
                 in_channels: int,
                 depth: int,
                 downscale: bool,
                 input_resolution: Tuple[int, int],
                 number_of_heads: int,
                 window_size: int = 7,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: Union[List[float], float] = 0.0,
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
        super(SwinTransformerStage, self).__init__()
        # Save parameters
        self.use_checkpoint: bool = use_checkpoint
        self.downscale: bool = downscale
        # Init downsampling
        self.downsample: nn.Module = PatchMerging(in_channels=in_channels) if downscale else nn.Identity()
        # Update resolution and channels
        self.input_resolution: Tuple[int, int] = (input_resolution[0] // 2, input_resolution[1] // 2) \
            if downscale else input_resolution
        in_channels = in_channels * 2 if downscale else in_channels
        # Get block
        block = DeformableSwinTransformerBlock if use_deformable_block else SwinTransformerBlock
        # Init blocks
        self.blocks: nn.ModuleList = nn.ModuleList([
            block(in_channels=in_channels,
                  input_resolution=self.input_resolution,
                  number_of_heads=number_of_heads,
                  window_size=window_size,
                  shift_size=0 if ((index % 2) == 0) else window_size // 2,
                  ff_feature_ratio=ff_feature_ratio,
                  dropout=dropout,
                  dropout_attention=dropout_attention,
                  dropout_path=dropout_path[index] if isinstance(dropout_path, list) else dropout_path,
                  sequential_self_attention=sequential_self_attention)
            for index in range(depth)])

    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update resolution
        self.input_resolution: Tuple[int, int] = (new_input_resolution[0] // 2, new_input_resolution[1] // 2) \
            if self.downscale else new_input_resolution
        # Update resolution of each block
        for block in self.blocks:  # type: SwinTransformerBlock
            block.update_resolution(new_window_size=new_window_size, new_input_resolution=self.input_resolution)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, 2 * channels, height // 2, width // 2]
        """
        # Downscale input tensor
        output: torch.Tensor = self.downsample(input)
        # Forward pass of each block
        for block in self.blocks:  # type: nn.Module
            # Perform checkpointing if utilized
            if self.use_checkpoint:
                output: torch.Tensor = checkpoint.checkpoint(block, output)
            else:
                output: torch.Tensor = block(output)
        return output
