from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    # NOTE: This is a custom BatchNorm2d layer.
    # -------------------------------------------------------------------------


    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()

        # NOTE: Because the layer is frozen, the parameters are stored as
        #       buffers, not as parameters.
        #----------------------------------------------------------------------

        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        
        # NOTE: Called when loading the model.
        #       Since the buffers are not trained, they are not saved in the
        #       state_dict, and thus need to be initialized here.
        #----------------------------------------------------------------------

        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # TODO: Normal BatchNorm2d layer
        # ---------------------------------------------------------------------

        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # NOTE: Freeze all layers except the last 3 if training the backbone,
        #       or freeze all layers except the last one if not training the
        # ---------------------------------------------------------------------
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # NOTE: if return_interm_layers is True, the output will be a dict
        #       containing the outputs of the layer 2, 3, and 4.
        #       Otherwise, the output will be a dict containing the output of
        #       the layer 4 (like normal ResNet).
        # ---------------------------------------------------------------------
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]

        # NOTE: `self.body` is just a wrapper around `backbone` that returns
        #       the output of the specified layers.
        # ---------------------------------------------------------------------
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        # NOTE:
        #   Output: a dict containing the output of the specified layers.
        # ---------------------------------------------------------------------
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            # NOTE: Interpolate the mask to the same size as the output of the
            #       corresponding layer.
            # -----------------------------------------------------------------
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d

        # NOTE: Get the ResNet model from torchvision.models. Dilation is disabled
        #       for the first two layers.
        # ---------------------------------------------------------------------
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)

        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    # NOTE: This layer is use to add the position encoding to the output of the
    #       backbone.
    # -------------------------------------------------------------------------

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        # NOTE: Compute the output of the backbone
        # ---------------------------------------------------------------------
        xs = self[0](tensor_list)

        out: List[NestedTensor] = []
        pos = []

        # NOTE: Sort the output of the backbone by the name of the layer.
        # ---------------------------------------------------------------------        
        for name, x in sorted(xs.items()):
            out.append(x)

        # NOTE: Compute the position encoding for each layer.
        # ---------------------------------------------------------------------
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        # NOTE: Return the output of the backbone and the position encoding.
        # ---------------------------------------------------------------------
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
