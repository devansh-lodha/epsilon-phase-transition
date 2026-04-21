import torch


class DeterministicAdamW:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_idx = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self) -> dict[str, float]:
        self.step_idx += 1
        p = self.params[0]
        if p.grad is None:
            raise RuntimeError("Gradient is None.")

        m, v = self.m[0], self.v[0]

        m.mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
        v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)

        bias_correction1 = 1 - self.beta1**self.step_idx
        bias_correction2 = 1 - self.beta2**self.step_idx

        m_hat = m / bias_correction1
        v_hat = v / bias_correction2

        p.data.mul_(1 - self.lr * self.weight_decay)
        denom = v_hat.sqrt().add_(self.eps)
        p.data.addcdiv_(m_hat, denom, value=-self.lr)

        rho_t = (v_hat.sqrt() / self.eps).mean().item()
        return {"rho_t": rho_t}


class DeterministicSGD:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-3,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]

    def step(self) -> dict[str, float]:
        p = self.params[0]
        if p.grad is None:
            raise RuntimeError("Gradient is None.")

        m = self.m[0]
        m.mul_(self.momentum).add_(p.grad, alpha=1 - self.momentum)

        p.data.mul_(1 - self.lr * self.weight_decay)
        p.data.add_(m, alpha=-self.lr)
        return {"rho_t": 0.0}  # Pure L2


class DeterministicSignGD:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 1e-3,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]

    def step(self) -> dict[str, float]:
        p = self.params[0]
        if p.grad is None:
            raise RuntimeError("Gradient is None.")

        m = self.m[0]
        m.mul_(self.momentum).add_(p.grad, alpha=1 - self.momentum)

        p.data.mul_(1 - self.lr * self.weight_decay)
        p.data.add_(torch.sign(m), alpha=-self.lr)
        return {"rho_t": float("inf")}  # Pure Linf
