import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .denoise_layer import DenoiseLayer


def count_conv2d_layers(model):
    num = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            num += 1
        else:
            num += count_conv2d_layers(module)
    return num


class DenoiseConv2d(DenoiseLayer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        T: int,
        timesteps: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        teacher: Optional[nn.Module] = None,   # 教师网络
        distill_alpha: float = 1.0,            # 蒸馏损失权重
    ):
        super().__init__(input_dim, output_dim, timesteps, hidden_dim=2048)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dm = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        # 拷贝预训练权重
        with torch.no_grad():
            self.dm.weight.copy_(weight)
            if bias is not None:
                self.dm.bias.copy_(bias)

        self.T = torch.tensor(T)
        self.teacher = teacher
        self.distill_alpha = distill_alpha

        self.weight = nn.Parameter(weight)
        if bias is None or bias.shape[0] == 0:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias)

        self.ploss = 0.0

    # -------- 蒸馏 p_losses ----------
    def p_losses(self, x_start, t, noise=None, loss_type="l1 + l2"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = self.pred_noise(x_noisy, t)   # student 输出

        # 原始噪声预测损失
        if loss_type == "l1":
            loss = F.l1_loss(noise, pred_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, pred_noise)
        elif loss_type == "l1 + l2":
            loss = F.l1_loss(noise, pred_noise) + F.mse_loss(noise, pred_noise)
        else:
            raise NotImplementedError(f"Unknown loss_type {loss_type}")

        # 蒸馏损失
        if self.teacher is not None:
            with torch.no_grad():
                teacher_out = self.teacher(x_noisy)
            distill_loss = F.mse_loss(pred_noise, teacher_out)
            loss = loss + self.distill_alpha * distill_loss

        return x_noisy, loss
    # ---------------------------------

    def _ct_1(self, t):
        return (1 - self.alphas[t]) / (
            self.sqrt_one_minus_alphas_cumprod[t] * self.sqrt_alphas[t]
        )

    def _ct_2(self, t):
        return self.sqrt_betas[t]

    def _ct_3(self, t):
        return 1.0 / self.sqrt_alphas[t]
        
    def freeze_teacher(model: nn.Module):
        """冻结所有 DenoiseConv2d 的 teacher，只训练 student (dm)"""
        for m in model.modules():
            if isinstance(m, DenoiseConv2d) and m.teacher is not None:
                for p in m.teacher.parameters():
                    p.requires_grad = False
                for p in m.dm.parameters():
                    p.requires_grad = True


    def freeze_student(model: nn.Module):
        """冻结所有 DenoiseConv2d 的 student (dm)，只训练 teacher"""
        for m in model.modules():
            if isinstance(m, DenoiseConv2d) and m.teacher is not None:
                for p in m.teacher.parameters():
                    p.requires_grad = True
                for p in m.dm.parameters():
                    p.requires_grad = False


    def fuse_weight(self):
        new_weight = self.weight - self._ct_1(self.T) * F.conv2d(
            self.weight, self.dm.weight, self.dm.bias,
            stride=self.dm.stride, padding=self.dm.padding
        )
        return new_weight

    def fuse_bias(self):
        new_bias = None
        if self.bias is not None:
            new_bias = self._ct_2(self.T) * self._ct_3(self.T) * self.bias
        return new_bias

    def fuse_parameters(self):
        self.weight, self.bias = nn.Parameter(self.fuse_weight()), nn.Parameter(
            self.fuse_bias()
        )

    def get_ploss(self):
        loss = self.ploss
        self.ploss = 0
        return loss

    def forward(self, x, t=None): 
        if t is None: 
            t = torch.full((x.shape[0],), self.T, device=x.device, dtype=torch.long) 
        weight, bias = self.weight, self.bias 
        if self.training: 
            x_noise, p_loss = self.p_losses(x, t) 
            self.ploss += p_loss 
            if not self.dm.weight.requires_grad: 
                weight = self.fuse_weight() 
                bias = self.fuse_bias()  
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


# ----------------- 转换函数 -----------------
def convert_denoise_conv2d(
    model: nn.Module,
    layer_num: int,
    timesteps: int = 1000,
    teacher_dict: Optional[dict] = None,
):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and (module.kernel_size[0] % 2 == 1):
            teacher = None
            if teacher_dict is not None and name in teacher_dict:
                teacher = teacher_dict[name]

            setattr(
                model,
                name,
                DenoiseConv2d(
                    input_dim=module.in_channels,
                    output_dim=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    T=layer_num,
                    timesteps=timesteps,
                    weight=module.weight.data.clone(),
                    bias=module.bias.data.clone() if module.bias is not None else None,
                    teacher=teacher,
                ),
            )
            layer_num -= 1
        else:
            convert_denoise_conv2d(module, layer_num, timesteps, teacher_dict)
    return model

