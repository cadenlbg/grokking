# grok/optimizer.py
import torch
from torch.optim import Optimizer
import numpy as np

# 自定义AdamW优化器 继承PyTorch原生基类
# 核心扩展特性：4种权重衰减策略切换 + 梯度噪声注入 + 标准AdamW核心逻辑 适配算术任务训练需求
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
        # 初始化默认超参配置 传递给父类Optimizer
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
        # 执行闭包计算损失 无则返回None
        loss = closure() if closure is not None else None

        # 遍历所有参数组 逐参数执行梯度更新逻辑
        for group in self.param_groups:
            for p in group["params"]:
                # 无梯度的参数跳过更新
                if p.grad is None:
                    continue
                grad = p.grad

                # 权重衰减形式1: honest 直接对梯度做权重衰减 梯度 += 权重衰减系数 * 参数值
                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "honest":
                        grad = grad + group["weight_decay"] * p.detach()

                # 初始化参数状态缓存 存储迭代步数/一阶矩/二阶矩/初始参数值
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if group["weight_decay_form"] == "to_init":
                        state["init"] = p.detach().clone()

                # 权重衰减核心逻辑 支持4种差异化策略 按需切换
                if group["weight_decay"] > 0:
                    # to_zero: 经典权重衰减 向0值收缩 p *= (1 - lr*wd)
                    if group["weight_decay_form"] == "to_zero":
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    # to_init: 向参数初始化值收缩 缓解过拟合 p += (初始值-p) * lr*wd
                    elif group["weight_decay_form"] == "to_init":
                        p.add_((state["init"] - p) * (group["lr"] * group["weight_decay"]))
                    # jiggle: 随机抖动衰减 乘高斯随机因子 提升泛化性
                    elif group["weight_decay_form"] == "jiggle":
                        p.mul_(torch.exp(torch.randn(1).to(p.device) * (group["lr"] * group["weight_decay"])))

                # Adam原生核心逻辑 一阶矩/二阶矩指数移动平均计算
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差修正 消除初始阶段的偏差影响
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                # 计算最终更新量 可选注入梯度高斯噪声 提升模型泛化能力
                upd = exp_avg / denom
                if group["noise_factor"] > 0:
                    upd += torch.randn_like(upd) * group["noise_factor"]
                p.add_(-step_size * upd)

        return loss

# 自定义SGD优化器 继承PyTorch原生基类
# 核心扩展特性：支持外部调节批次大小+梯度归一化+Nesterov动量+标准权重衰减
class CustomSGD(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        batchsize=32,
        momentum=0.0,
        weight_decay=0.0,
        nesterov=False
    ):
        # 入参合法性校验 杜绝无效超参传入
        if lr < 0.0:
            raise ValueError(f"无效的学习率：{lr}，必须≥0")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"无效的动量系数：{momentum}，必须在[0,1]之间")
        if weight_decay < 0.0:
            raise ValueError(f"无效的权重衰减：{weight_decay}，必须≥0")

        # 初始化默认超参配置
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

        # 遍历参数组执行更新
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            batchsize = group["batchsize"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # 按批次大小归一化梯度 适配不同batchsize的梯度幅度
                grad = p.grad / batchsize
                state = self.state[p]

                # 标准权重衰减 向0收缩
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # 初始化动量缓存 仅在动量系数>0时生效
                if len(state) == 0 and momentum > 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                # 动量更新逻辑 支持普通动量/Nesterov动量两种模式
                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    if nesterov:
                        grad = grad.add_(buf, alpha=momentum)
                    else:
                        grad = buf

                # 执行参数梯度更新
                p.add_(grad, alpha=-lr)

        return loss

# 自定义RMSprop优化器 继承PyTorch原生基类
# 核心扩展特性：支持批次大小梯度归一化+梯度平方移动平均+数值稳定项+权重衰减
class CustomRMSprop(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        batchsize=32,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0.0
    ):
        # 入参合法性校验
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

                # 初始化状态缓存 存储梯度平方的移动平均值
                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(p)

                square_avg = state["square_avg"]

                # 标准权重衰减逻辑
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # RMSprop核心逻辑：梯度平方的指数移动平均 更新
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                # 带数值稳定项的参数更新 避免分母为0
                p.addcdiv_(grad, (square_avg.sqrt() + eps), value=-lr)

        return loss

# 纯动量优化器 继承PyTorch原生基类
# 核心特性：专属阻尼系数+批次梯度归一化+权重衰减 区别于SGD的动量实现 更轻量化
class CustomMomentum(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        batchsize=32,
        momentum=0.9,
        dampening=0.0,
        weight_decay=0.0
    ):
        # 入参合法性校验
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

                # 标准权重衰减逻辑
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # 初始化动量缓存 首次更新时创建
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                # 带阻尼的动量核心更新逻辑 阻尼系数缓解动量过大的震荡问题
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                # 执行参数更新
                p.add_(buf, alpha=-lr)

        return loss