# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

def gumbel_softmax(probs, eps=1e-10, tau=1, mode="hard"):
    probs = probs.log()
    samples = torch.rand_like(probs).to(probs.device)
    gumbel_samples = -torch.log(-torch.log(samples + eps) + eps)
    gumbel_samples = (probs + gumbel_samples) / tau
    y_soft = gumbel_samples.softmax(-1)
    if mode == "soft":
        return y_soft, y_soft.argmax(-1).view(probs.shape[0], -1)
    index = y_soft.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(probs).scatter_(-1, index, 1.0)
    y_hard = y_hard - y_soft.detach() + y_soft
    return y_hard, y_soft

class PositionwiseFeedForwardMoE(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, num_experts, capacity_factor, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForwardMoE, self).__init__()

        self.experts = repeat(
            num_experts,
            lambda lnum: PositionwiseFeedForward(
                idim,
                hidden_units // num_experts,
                dropout_rate,
                activation
            ),
        )
        self.expert_predictor = torch.nn.Linear(idim, num_experts)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
    def forward(self, x):
        """Forward function."""
        bs, ts, dim = x.shape
        x = x.view(-1, dim)
        capacity = int(self.capacity_factor * bs * ts / self.num_experts)
        
        probs = self.softmax(self.expert_predictor(x))
        expert_probs, expert_chosen = torch.topk(probs, 1, -1)
        final_output = x.new_zeros(x.shape)
        
        expert_indexes_list = []
        for expert_idx in range(self.num_experts):
            indices = torch.eq(expert_chosen, expert_idx).nonzero(as_tuple=True)[0]
            outputs = self.experts[expert_idx](x[indices, :])
            final_output[indices, :] = outputs
        final_output = final_output * (expert_probs / expert_probs.detach()).view(-1, 1)
        final_output = final_output.view(bs, ts, dim)
        return final_output, probs.view(bs, ts, -1), expert_chosen.view(bs, -1)

class PositionwiseFeedForwardUttMoE(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
                self, 
                idim, 
                hidden_units, 
                num_experts, 
                capacity_factor, 
                dropout_rate, 
                activation=torch.nn.ReLU(), 
                gs=False, 
                gs_mode=None, 
                tau=None, 
                split_dims=False,
                use_expert_predictor=True,
                global_expert=False,
                global_expert_dim=512
                ):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForwardUttMoE, self).__init__()

        self.experts = repeat(
            num_experts,
            lambda lnum: PositionwiseFeedForward(
                idim,
                hidden_units // num_experts if split_dims else hidden_units[lnum],
                dropout_rate,
                activation,
                idim // 2 if global_expert is not False else idim,
            ),
        )
        if use_expert_predictor:
            self.expert_predictor = torch.nn.Linear(idim, num_experts)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gs = gs
        self.gs_mode = gs_mode
        self.tau = tau
        if gs:
            assert gs_mode in ["soft"], f'{gs_mode}'
        self.global_expert = None
        if global_expert:
            self.global_expert = PositionwiseFeedForward(
                idim,
                global_expert_dim,
                dropout_rate,
                activation,
                idim // 2,
            )
        
    def forward(self, x, change_expert=None, hard_gs_decoding=False, custom_expert_chosen=None):
        """Forward function."""
        bs, ts, dim = x.shape      
        if custom_expert_chosen is None:      
            probs = self.softmax(self.expert_predictor(torch.mean(x, 1)))
        else:
            probs = torch.full((bs, 1), fill_value=1.0, device=x.device)
            
        if self.gs:
            probs, expert_chosen = gumbel_softmax(probs, mode=self.gs_mode, tau=self.tau)
            if hard_gs_decoding:
                argmax_out = probs.argmax(-1)
                if change_expert is not None:
                    argmax_out = torch.full(argmax_out.shape, fill_value=change_expert, device=argmax_out.device, dtype=argmax_out.dtype)
            if self.gs_mode == "soft":
                all_expert_outputs = []
                for expert_idx in range(self.num_experts):
                    
                    if hard_gs_decoding:
                        if expert_idx != argmax_out.squeeze().item():
                            continue
                        # final_output = self.experts[expert_idx](x) / 2
                        final_output = self.experts[expert_idx](x)
                        
                    else:
                        outputs = self.experts[expert_idx](x)
                        outputs = outputs * probs[:, expert_idx].view(-1, 1, 1)
                        all_expert_outputs.append(outputs)
                        
                if not hard_gs_decoding:
                    # final_output = torch.stack(all_expert_outputs, -1).mean(-1)
                    final_output = torch.stack(all_expert_outputs, -1).sum(-1)
        else:
            if custom_expert_chosen is None:
                expert_probs, expert_chosen = torch.topk(probs, 1, -1)
                
            else:
                expert_chosen = custom_expert_chosen
            
            if change_expert is not None:
                expert_chosen = torch.full(expert_chosen.shape, fill_value=change_expert, device=expert_chosen.device, dtype=expert_chosen.dtype)
            
            if self.global_expert:
                final_output = x.new_zeros(x.shape[0], x.shape[1], x.shape[2] // 2)
            else:
                final_output = x.new_zeros(x.shape)
            expert_indexes_list = []    
            for expert_idx in range(self.num_experts):
                indices = torch.eq(expert_chosen, expert_idx).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    outputs = self.experts[expert_idx](x[indices, :])
                    final_output[indices, :] = outputs
            
            if custom_expert_chosen is None:
                final_output = final_output * (expert_probs / expert_probs.detach()).view(-1, 1, 1)
            
        if self.global_expert:
            global_expert_vals = self.global_expert(x)
            final_output = torch.cat([global_expert_vals, final_output], -1)
            

        return final_output, probs.view(bs, -1), expert_chosen.view(bs, -1)
    
    def bs_decoder(self, x, local_bw, expert_history, expert_idx_history, residual):
        bs, ts, dim = x.shape            
        probs = self.softmax(self.expert_predictor(torch.mean(x, 1)))
        probs, expert_chosen = gumbel_softmax(probs, mode=self.gs_mode, tau=self.tau)
        num_experts = probs.shape[-1]
        local_topk_out = torch.topk(probs, min(local_bw, num_experts), -1)
        local_topk_values = torch.log(local_topk_out.values.squeeze())
        local_topk_indices = local_topk_out.indices.squeeze().int().tolist()
        

        rename = False
        if expert_history is None:
            expert_history = [local_topk_values]
            expert_idx_history = [[(idx, x) for idx, x in enumerate(local_topk_indices)]]
            x = x.repeat(min(local_bw, num_experts), 1, 1)
            residual = residual.repeat(min(local_bw, num_experts), 1, 1)
        else:
            global_values = torch.stack([torch.add(x, y) for x in expert_history[-1] for y in local_topk_values[-1]], 0)
            global_ids = [(expert_idx_history[-1][x][0], local_topk_indices[-1][y]) for x in range(len(expert_history[-1])) for y in range(len(local_topk_values[-1]))]
            
            global_topk_out = torch.topk(global_values, min(local_bw, len(global_values)), -1)
            global_topk_indices = global_topk_out.indices.int().tolist()
            global_topk_values = global_topk_out.values
            selected_ids = [global_ids[x] for x in global_topk_indices]
            expert_idx_history.append(selected_ids)
            expert_history.append(global_topk_values)
            rename = True
        res_ = []
        residual_ = []
        for idx in expert_idx_history[-1]:
            if idx == -1: continue
            x_idx, expert_idx = idx
            residual_.append(residual[x_idx])
            res = self.experts[expert_idx](x[x_idx])
            res_.append(res)
        if rename:
            expert_idx_history[-1] = [(idx, x) for idx, x in enumerate(global_topk_indices)]
        residual = torch.stack(residual_, 0)
        final_output = torch.stack(res_, 0)
        return final_output, expert_history, expert_idx_history, residual
    
class ConformerMoEEncoderLayer(torch.nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
        macaron_moe=False,
        moe=False,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        self.macaron_moe = macaron_moe
        self.moe = moe
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, dialectid=None, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # whether to use macaron style
        macaron_expert_probs, macaron_expert_chosen = [], []
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            if self.macaron_moe:
                x, macaron_expert_probs, macaron_expert_chosen = self.feed_forward_macaron(x)
            else:
                x = self.feed_forward_macaron(x)
            
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(x)
            
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        
        if self.normalize_before:
            x = self.norm_ff(x)
        if self.moe:
            if dialectid is not None:
                dialectid = dialectid.squeeze() - 1
            x, expert_probs, expert_chosen = self.feed_forward(x, custom_expert_chosen=dialectid) 
        else:
            x = self.feed_forward(x) 
            expert_probs, expert_chosen = [], []
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(x)
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask, (macaron_expert_probs, macaron_expert_chosen), (expert_probs, expert_chosen)

        return x, mask, (macaron_expert_probs, macaron_expert_chosen), (expert_probs, expert_chosen)

    def bs_decoder(self, x_input, mask, local_bw, cache=None, expert_history=None, expert_idx_history=None):
        
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        stoch_layer_coeff = 1.0

        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = self.feed_forward_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(x)
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
            
        x_q = x

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
            
        if not self.normalize_before:
            x = self.norm_mha(x)

        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
            
        x, expert_history, expert_idx_history, residual = self.feed_forward.bs_decoder(x, local_bw=local_bw, expert_history=expert_history, expert_idx_history=expert_idx_history, residual=residual) 
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(x)
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask, expert_history, expert_idx_history

        return x, mask, (macaron_expert_probs, macaron_expert_chosen), (expert_history, expert_idx_history)
    
        
    
class ConformerEncoderMoe(AbsEncoder):
    """Conformer encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: any = 2048,
        macaron_style_linear_units: int = 1024,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        macaron_positionwise_layer_type: str = "linear",
        num_experts: int = 1,
        capacity_factor: float = 1.0, 
        use_load_balancing_loss: bool = True,
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        gs: bool = False,
        gs_mode: str = None,
        tau: float = None,
        use_dialect_id: bool = False,
        use_expert_predictor: bool = True,
        global_expert: bool = False,
        global_expert_dim: int = 512
        
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        self.moe = False
        self.use_load_balancing_loss = use_load_balancing_loss
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "moe":
            self.moe = True
            self.num_experts = num_experts
            assert linear_units % num_experts == 0, f'linear_units ({linear_units}) must be multiple by num_experts({num_experts})'
            positionwise_layer = PositionwiseFeedForwardMoE
            positionwise_layer_args = (
                output_size,
                linear_units,
                num_experts,
                capacity_factor,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "guided_utt_moe":
            self.moe = True
            self.num_experts = num_experts
            positionwise_layer = PositionwiseFeedForwardUttMoE
            positionwise_layer_args = (
                output_size,
                linear_units,
                num_experts,
                capacity_factor,
                dropout_rate,
                activation,
                False,
                gs_mode,
                tau,
                False,
                use_expert_predictor,
                global_expert,
                global_expert_dim,
                
            )
        elif positionwise_layer_type == "utt_moe":
            self.moe = True
            self.num_experts = num_experts
            assert linear_units % num_experts == 0, f'linear_units ({linear_units}) must be multiple by num_experts({num_experts})'
            positionwise_layer = PositionwiseFeedForwardUttMoE
            positionwise_layer_args = (
                output_size,
                linear_units,
                num_experts,
                capacity_factor,
                dropout_rate,
                activation,
                gs,
                gs_mode,
                tau,
                True,
                use_expert_predictor
            )
        elif positionwise_layer_type == "utt_mix_moe":
            self.moe = True
            self.num_experts = num_experts
            positionwise_layer = PositionwiseFeedForwardUttMoE
            positionwise_layer_args = (
                output_size,
                linear_units,
                num_experts,
                capacity_factor,
                dropout_rate,
                activation,
                gs,
                gs_mode,
                tau,
                False,
                use_expert_predictor
                )
        else:
            raise NotImplementedError("Support only linear or moe.")
        self.macaron_moe = False
        if macaron_positionwise_layer_type == "linear":
            positionwise_layer_macaron = PositionwiseFeedForward
            positionwise_layer_macaron_args = (
                output_size,
                macaron_style_linear_units,
                dropout_rate,
                activation,
            )
        elif macaron_positionwise_layer_type == "moe":
            self.macaron_moe = True
            
            assert macaron_style_linear_units % num_experts == 0, f'macaron_style_linear_units ({macaron_style_linear_units}) must be multiple by num_experts({num_experts})'
            positionwise_layer_macaron = PositionwiseFeedForwardMoE
            positionwise_layer_macaron_args = (
                output_size,
                macaron_style_linear_units,
                num_experts,
                capacity_factor,
                dropout_rate,
                activation,
            )
        else:
            raise NotImplementedError("Support only linear or moe.")
    

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: ConformerMoEEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer_macaron(*positionwise_layer_macaron_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate[lnum],
                macaron_moe=self.macaron_moe,
                moe=self.moe,
            ),
            layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

    def output_size(self) -> int:
        return self._output_size

    def bs_decoder(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        bw,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)
        
        intermediate_outs = []
        all_macaron_expert_probs = []
        all_macaron_expert_chosen = []
        all_expert_probs = []
        all_expert_chosen = []
        expert_idx_history = None
        expert_history = None
        for layer_idx, encoder_layer in enumerate(self.encoders):
            xs_pad, masks, expert_history, expert_idx_history = encoder_layer.bs_decoder(xs_pad, masks, local_bw=bw, expert_idx_history=expert_idx_history, expert_history=expert_history)
        
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None, (all_macaron_expert_probs, all_macaron_expert_chosen, all_expert_probs, all_expert_chosen)
    
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        dialectid=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []
        all_macaron_expert_probs = []
        all_macaron_expert_chosen = []
        all_expert_probs = []
        all_expert_chosen = []
            # xs_pad, masks = self.encoders(xs_pad, masks)
        for layer_idx, encoder_layer in enumerate(self.encoders):
            xs_pad, masks, (macaron_expert_probs, macaron_expert_chosen), (expert_probs, expert_chosen) = encoder_layer(xs_pad, masks, dialectid=dialectid)
            all_macaron_expert_probs.append(macaron_expert_probs)
            all_macaron_expert_chosen.append(macaron_expert_chosen)
            all_expert_probs.append(expert_probs)
            all_expert_chosen.append(expert_chosen)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None, (all_macaron_expert_probs, all_macaron_expert_chosen), (all_expert_probs, all_expert_chosen)
        return xs_pad, olens, None, (all_macaron_expert_probs, all_macaron_expert_chosen, all_expert_probs, all_expert_chosen)

