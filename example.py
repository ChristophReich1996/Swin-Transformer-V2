from typing import List

import torch

from swin_transformer_v2 import swin_transformer_v2_t, SwinTransformerV2


def main() -> None:
    # Make input tensor and init Swin Transformer V2, for the custom deformable version set use_deformable_block=True
    input = torch.rand(2, 3, 256, 256)
    swin_transformer: SwinTransformerV2 = swin_transformer_v2_t(in_channels=3,
                                                                window_size=8,
                                                                input_resolution=(256, 256),
                                                                sequential_self_attention=False,
                                                                use_checkpoint=False)
    # Perform forward pass
    features: List[torch.Tensor] = swin_transformer(input)
    # Print shape of features
    for feature in features:
        print(feature.shape)

    # Update the resolution and window size of the Swin Transformer V2 and init new input
    swin_transformer.update_resolution(new_window_size=16, new_input_resolution=(512, 512))
    input = torch.rand(2, 3, 512, 512)
    # Perform forward pass
    features: List[torch.Tensor] = swin_transformer(input)
    # Print shape of features
    for feature in features:
        print(feature.shape)


if __name__ == '__main__':
    main()
