import torch
import torch.nn.functional as F
from typeguard import check_argument_types


class CTC(torch.nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or gtnctc
        reduce: reduce the CTC loss into a scalar
        ignore_nan_grad: Same as zero_infinity (keeping for backward compatiblity)
        zero_infinity:  Whether to zero infinite losses and the associated gradients.
    """

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = None,
        zero_infinity: bool = True,
        num_decoders: int = 1,
    ):
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.num_decoders = num_decoders
        if num_decoders == 1:
            self.ctc_lo = torch.nn.Linear(eprojs, odim)
        else:
            ctc_lo = torch.nn.ModuleList()
            for idx in range(num_decoders):
                ctc_lo.append(torch.nn.Linear(eprojs, odim))
            self.ctc_lo = ctc_lo
            
        self.ctc_type = ctc_type
        if ignore_nan_grad is not None:
            zero_infinity = ignore_nan_grad

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(
                reduction="none", zero_infinity=zero_infinity
            )

        elif self.ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply
        else:
            raise ValueError(f'ctc_type must be "builtin" or "gtnctc": {self.ctc_type}')

        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        if self.ctc_type == "builtin":
            if self.num_decoders == 1:
                th_pred = th_pred.log_softmax(2)
                loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
                size = th_pred.size(1)
            else:
                _th_pred = []
                for idx in range(self.num_decoders):
                    th_pred_ = th_pred[idx].softmax(2)
                    _th_pred.append(th_pred_)
                th_pred = torch.stack(_th_pred, -1) 
                th_pred = torch.mean(th_pred, -1) 
                th_pred = th_pred.log()
                size = th_pred.size(1)
                loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        
        if self.num_decoders == 1:
            ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
            device = hs_pad.device
            dtype = hs_pad.dtype
        else:
            ys_hat_ = []
            for idx in range(len(self.ctc_lo)):
                _ys_hat = self.ctc_lo[idx](F.dropout(hs_pad[idx], p=self.dropout_rate))
                ys_hat_.append(_ys_hat)
            device = _ys_hat.device
            dtype = _ys_hat.dtype
            ys_hat = ys_hat_
            
    
        if self.ctc_type == "gtnctc":
            # gtn expects list form for ys
            ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
        else:
            # ys_hat: (B, L, D) -> (L, B, D)
            if self.num_decoders == 1:
                ys_hat = ys_hat.transpose(0, 1)
            else:
                ys_hat_ = []
                for idx in range(len(self.ctc_lo)):
                    _ys_hat = ys_hat[idx].transpose(0, 1)
                    ys_hat_.append(_ys_hat)
                ys_hat = ys_hat_
            # (B, L) -> (BxL,)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=device, dtype=dtype
        )

        return loss

    def softmax(self, hs_pad):
        """softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad, unsqueeze_later=False):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        
        if self.num_decoders == 1:
            return F.log_softmax(self.ctc_lo(hs_pad), dim=2)
        ys_hat_ = []
        for idx in range(len(self.ctc_lo)):
            if unsqueeze_later:
                x = hs_pad[idx].unsqueeze(0)
            else:
                x = hs_pad[idx]
            _ys_hat = self.ctc_lo[idx](x).softmax(2)
            ys_hat_.append(_ys_hat)
        ys_hat = torch.stack(ys_hat_, -1)
        ys_hat = torch.mean(ys_hat, -1)
        return ys_hat.log()
    
    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        if self.num_decoders == 1:
            return torch.argmax(self.ctc_lo(hs_pad), dim=2)
                
        ys_hat_ = []
        for idx in range(len(self.ctc_lo)):
            _ys_hat = self.ctc_lo[idx](hs_pad[idx]).softmax(2)
            ys_hat_.append(_ys_hat)
        ys_hat = torch.stack(ys_hat_, -1)
        ys_hat = torch.mean(ys_hat, -1)
        return torch.argmax(ys_hat, dim=2)
