# grok/optimizer.py
import torch
from torch.optim import Optimizer
import numpy as np

# ========== 原有 CustomAdamW（从 training.py 迁移过来） ==========
class CustomAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        noise_factor=0.0,
        weight_decay_form="to_zero",
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            noise_factor=noise_factor,
            weight_decay_form=weight_decay_form,
        )
        super(CustomAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # 权重衰减
                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "honest":
                        grad = grad + group["weight_decay"] * p.detach()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if group["weight_decay_form"] == "to_init":
                        state["init"] = p.detach().clone()

                # 权重衰减逻辑
                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "to_zero":
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    elif group["weight_decay_form"] == "to_init":
                        p.add_((state["init"] - p) * (group["lr"] * group["weight_decay"]))
                    elif group["weight_decay_form"] == "jiggle":
                        p.mul_(torch.exp(torch.randn(1).to(p.device) * (group["lr"] * group["weight_decay"])))

                # Adam 核心逻辑
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                # 梯度噪声
                upd = exp_avg / denom
                if group["noise_factor"] > 0:
                    upd += torch.randn_like(upd) * group["noise_factor"]
                p.add_(-step_size * upd)

        return loss

# ========== 原有 CustomSGD（保持不变） ==========
class CustomSGD(Optimizer):
    """
    自定义SGD优化器（支持调节lr、batchsize、momentum、weight_decay等参数）
    :param batchsize: 批次大小（用于适配梯度更新策略，可外部调节）
    :param lr: 学习率
    :param momentum: 动量系数（0=无动量，1=最大动量）
    :param weight_decay: 权重衰减系数
    :param nesterov: 是否使用Nesterov动量
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        batchsize=32,  # 可调节批次大小
        momentum=0.0,
        weight_decay=0.0,
        nesterov=False
    ):
        # 参数校验
        if lr < 0.0:
            raise ValueError(f"无效的学习率：{lr}，必须≥0")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"无效的动量系数：{momentum}，必须在[0,1]之间")
        if weight_decay < 0.0:
            raise ValueError(f"无效的权重衰减：{weight_decay}，必须≥0")

        # 默认参数配置
        defaults = dict(
            lr=lr,
            batchsize=batchsize,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super(CustomSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            batchsize = group["batchsize"]  # 使用自定义batchsize参数

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad / batchsize  # 按批次大小归一化梯度
                state = self.state[p]

                # 权重衰减
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # 初始化动量缓存
                if len(state) == 0 and momentum > 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                # 动量更新逻辑
                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)  # 动量缓存更新
                    if nesterov:
                        grad = grad.add_(buf, alpha=momentum)  # Nesterov动量
                    else:
                        grad = buf

                # 参数更新
                p.add_(grad, alpha=-lr)

        return loss

# ========== 原有 CustomRMSprop（保持不变） ==========
class CustomRMSprop(Optimizer):
    """
    自定义RMSprop优化器（支持调节lr、batchsize、alpha、eps、weight_decay等参数）
    :param alpha: 移动平均衰减系数
    :param eps: 数值稳定项
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        batchsize=32,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0.0
    ):
        if lr < 0.0:
            raise ValueError(f"无效的学习率：{lr}，必须≥0")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"无效的alpha：{alpha}，必须在[0,1]之间")
        if eps < 0.0:
            raise ValueError(f"无效的eps：{eps}，必须≥0")
        if weight_decay < 0.0:
            raise ValueError(f"无效的权重衰减：{weight_decay}，必须≥0")

        defaults = dict(
            lr=lr,
            batchsize=batchsize,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay
        )
        super(CustomRMSprop, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            batchsize = group["batchsize"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad / batchsize
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(p)

                square_avg = state["square_avg"]

                # 权重衰减
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # RMSprop核心逻辑：梯度平方的移动平均
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                # 参数更新（加入数值稳定项）
                p.addcdiv_(grad, (square_avg.sqrt() + eps), value=-lr)

        return loss

# ========== 原有 CustomMomentum（保持不变） ==========
class CustomMomentum(Optimizer):
    """
    纯动量优化器（momentum-based，支持调节lr、batchsize、momentum、dampening等参数）
    :param dampening: 动量阻尼系数
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        batchsize=32,
        momentum=0.9,
        dampening=0.0,
        weight_decay=0.0
    ):
        if lr < 0.0:
            raise ValueError(f"无效的学习率：{lr}，必须≥0")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"无效的动量系数：{momentum}，必须在[0,1]之间")
        if dampening < 0.0 or dampening > 1.0:
            raise ValueError(f"无效的阻尼系数：{dampening}，必须在[0,1]之间")
        if weight_decay < 0.0:
            raise ValueError(f"无效的权重衰减：{weight_decay}，必须≥0")

        defaults = dict(
            lr=lr,
            batchsize=batchsize,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay
        )
        super(CustomMomentum, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            batchsize = group["batchsize"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad / batchsize
                state = self.state[p]

                # 权重衰减
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # 初始化动量缓存
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                # 动量更新（带阻尼）
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                # 参数更新
                p.add_(buf, alpha=-lr)

        return loss