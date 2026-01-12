import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class D3PMConfig:
    num_classes: int
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


class D3PMScheduler:
    """
    Multinomial / categorical diffusion with uniform corruption:

    Forward transition over K categories:
        Q_t = a_t I + (1 - a_t) * 1/K
    which corresponds to: keep with prob a_t, else replace with a uniform random label.
    """

    def __init__(
        self,
        num_classes: int,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ) -> None:
        self.K = int(num_classes)
        self.T = int(num_timesteps)
        if self.K <= 1:
            raise ValueError("num_classes must be >= 2.")
        if self.T <= 0:
            raise ValueError("num_timesteps must be >= 1.")

        betas = torch.linspace(float(beta_start), float(beta_end), self.T, device=device, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = torch.cat([torch.zeros(1, device=device, dtype=torch.float32), betas], dim=0)  # (T+1,)
        self.alphas = torch.cat([torch.ones(1, device=device, dtype=torch.float32), alphas], dim=0)  # (T+1,)
        self.alpha_bars = torch.cat([torch.ones(1, device=device, dtype=torch.float32), alpha_bars], dim=0)  # (T+1,)

        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[torch.Tensor] = None

    @property
    def config(self) -> D3PMConfig:
        beta_start = float(self.betas[1].item()) if self.T >= 1 else 1e-4
        beta_end = float(self.betas[-1].item()) if self.T >= 1 else 0.02
        return D3PMConfig(num_classes=self.K, num_timesteps=self.T, beta_start=beta_start, beta_end=beta_end)

    def to(self, device: str | torch.device) -> "D3PMScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        if self.timesteps is not None:
            self.timesteps = self.timesteps.to(device)
        return self

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> torch.Tensor:
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be >= 1.")
        if num_inference_steps >= self.T:
            timesteps = torch.arange(self.T, 0, -1, dtype=torch.long)
        else:
            timesteps = torch.linspace(float(self.T), 1.0, num_inference_steps, dtype=torch.float32).round().long()
            timesteps = torch.unique_consecutive(timesteps)
            if timesteps.numel() == 0 or timesteps[0].item() != self.T:
                timesteps = torch.cat([torch.tensor([self.T], dtype=torch.long), timesteps], dim=0)
            if timesteps[-1].item() != 1:
                timesteps = torch.cat([timesteps, torch.tensor([1], dtype=torch.long)], dim=0)
        if device is not None:
            timesteps = timesteps.to(device)
        self.timesteps = timesteps
        self.num_inference_steps = int(timesteps.numel())
        return timesteps

    @staticmethod
    def sample_from_logprobs(log_probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        log_probs: (B,H,W,K) categorical log-probabilities.
        Returns: (B,H,W) int64 samples.
        """
        if log_probs.ndim != 4:
            raise ValueError(f"Expected log_probs (B,H,W,K), got {tuple(log_probs.shape)}")
        u = torch.rand(log_probs.shape, device=log_probs.device, dtype=torch.float32, generator=generator)
        u = u.clamp_(min=1e-6, max=1.0 - 1e-6)
        g = -torch.log(-torch.log(u))
        return (log_probs.float() + g).argmax(dim=-1).to(dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0).

        x0: (B,H,W) int64 in [0..K-1]
        t:  (B,) int64 in [0..T]
        """
        if x0.ndim != 3:
            raise ValueError(f"Expected x0 (B,H,W), got {tuple(x0.shape)}")
        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError(f"Expected t (B,) matching batch, got {tuple(t.shape)} vs batch {x0.shape[0]}")
        a_bar = self.alpha_bars[t].view(-1, 1, 1).to(device=x0.device, dtype=torch.float32)
        keep = (torch.rand(x0.shape, device=x0.device, dtype=torch.float32, generator=generator) < a_bar).to(
            dtype=torch.long
        )
        rnd = torch.randint(0, self.K, size=x0.shape, device=x0.device, dtype=torch.long, generator=generator)
        return keep * x0.to(dtype=torch.long) + (1 - keep) * rnd

    def _log_Qbar_row(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        log q(x_t = i | x_0) for all i, per-pixel.
        Returns log-probs over i: (B,H,W,K).
        """
        if x0.ndim != 3:
            raise ValueError(f"Expected x0 (B,H,W), got {tuple(x0.shape)}")
        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError(f"Expected t (B,) matching batch, got {tuple(t.shape)} vs batch {x0.shape[0]}")
        a_bar = self.alpha_bars[t].view(-1, 1, 1).to(device=x0.device, dtype=torch.float32)
        # For fixed x0 label k, distribution over i:
        # p(i) = a_bar * 1[i==k] + (1 - a_bar)/K
        base = torch.log((1.0 - a_bar).clamp(min=1e-12)) - torch.log(
            torch.tensor(float(self.K), device=x0.device, dtype=torch.float32)
        )  # (B,1,1)
        diag = a_bar + (1.0 - a_bar) / float(self.K)  # (B,1,1)
        bump = torch.log(diag.clamp(min=1e-12)) - base  # (B,1,1)
        logp = base.unsqueeze(-1).expand(x0.shape[0], x0.shape[1], x0.shape[2], self.K).clone()
        bump_map = bump.expand(x0.shape[0], x0.shape[1], x0.shape[2]).unsqueeze(-1)  # (B,H,W,1)
        logp.scatter_add_(-1, x0.to(dtype=torch.long).unsqueeze(-1), bump_map)
        return logp

    def _log_Q_step_cols(self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor) -> torch.Tensor:
        """
        log q(x_t | x_{t_prev}=i) for all i, per-pixel, for the *effective* transition from t_prev -> t.

        Returns log-probs over i (previous state): (B,H,W,K) for fixed x_t.
        """
        if x_t.ndim != 3:
            raise ValueError(f"Expected x_t (B,H,W), got {tuple(x_t.shape)}")
        if t.ndim != 1 or t_prev.ndim != 1 or t.shape != t_prev.shape or t.shape[0] != x_t.shape[0]:
            raise ValueError(f"Expected t,t_prev (B,) matching batch, got t={tuple(t.shape)} t_prev={tuple(t_prev.shape)}")
        a_bar_t = self.alpha_bars[t].view(-1, 1, 1).to(device=x_t.device, dtype=torch.float32)
        a_bar_prev = self.alpha_bars[t_prev].view(-1, 1, 1).to(device=x_t.device, dtype=torch.float32)
        a_step = (a_bar_t / a_bar_prev.clamp(min=1e-12)).clamp(min=0.0, max=1.0)  # (B,1,1)
        base = torch.log((1.0 - a_step).clamp(min=1e-12)) - torch.log(
            torch.tensor(float(self.K), device=x_t.device, dtype=torch.float32)
        )  # (B,1,1)
        diag = a_step + (1.0 - a_step) / float(self.K)  # (B,1,1)
        bump = torch.log(diag.clamp(min=1e-12)) - base  # (B,1,1)
        logp = base.unsqueeze(-1).expand(x_t.shape[0], x_t.shape[1], x_t.shape[2], self.K).clone()
        bump_map = bump.expand(x_t.shape[0], x_t.shape[1], x_t.shape[2]).unsqueeze(-1)  # (B,H,W,1)
        logp.scatter_add_(-1, x_t.to(dtype=torch.long).unsqueeze(-1), bump_map)
        return logp

    def q_posterior_logprobs(
        self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, t_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        log q(x_{t_prev} | x_t, x_0) as (B,H,W,K) over x_{t_prev}.
        """
        if t_prev is None:
            t_prev = (t - 1).clamp(min=0)
        if torch.any(t_prev < 0) or torch.any(t_prev > t):
            raise ValueError("t_prev must satisfy 0 <= t_prev <= t elementwise.")
        # If t_prev==0, x_{t_prev} == x0 deterministically.
        if torch.all(t_prev == 0):
            out = torch.full((x0.shape[0], x0.shape[1], x0.shape[2], self.K), -float("inf"), device=x0.device)
            out.scatter_(-1, x0.to(dtype=torch.long).unsqueeze(-1), 0.0)
            return out

        log_Qt = self._log_Q_step_cols(x_t, t=t, t_prev=t_prev)  # (B,H,W,K)
        log_Qbar_prev = self._log_Qbar_row(x0, t_prev)  # (B,H,W,K)
        logits = log_Qt + log_Qbar_prev
        return logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    def p_theta_posterior_logprobs(
        self, x_t: torch.Tensor, t: torch.Tensor, logits_x0: torch.Tensor, t_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        log p_theta(x_{t_prev} | x_t) as (B,H,W,K) over x_{t_prev}, where
            p_theta(x_{t_prev} | x_t) = sum_{x0} q(x_{t_prev} | x_t, x0) p_theta(x0 | x_t).

        logits_x0: (B,K,H,W) network logits for x0.
        """
        if logits_x0.ndim != 4:
            raise ValueError(f"Expected logits_x0 (B,K,H,W), got {tuple(logits_x0.shape)}")
        if t_prev is None:
            t_prev = (t - 1).clamp(min=0)
        if torch.any(t_prev < 0) or torch.any(t_prev > t):
            raise ValueError("t_prev must satisfy 0 <= t_prev <= t elementwise.")

        # t_prev == 0 => x_{t_prev} is x0, so posterior reduces to p_theta(x0|x_t).
        if torch.all(t_prev == 0):
            return torch.log_softmax(logits_x0.float(), dim=1).permute(0, 2, 3, 1)

        p_x0 = torch.softmax(logits_x0.permute(0, 2, 3, 1).float(), dim=-1)  # (B,H,W,K)
        log_Qt = self._log_Q_step_cols(x_t, t=t, t_prev=t_prev)  # (B,H,W,K) over i

        a_bar = self.alpha_bars[t_prev].view(-1, 1, 1).to(device=x_t.device, dtype=torch.float32)  # (B,1,1)
        base = torch.log((1.0 - a_bar).clamp(min=1e-12)) - torch.log(
            torch.tensor(float(self.K), device=x_t.device, dtype=torch.float32)
        )  # (B,1,1)
        diag = a_bar + (1.0 - a_bar) / float(self.K)  # (B,1,1)
        bump = torch.log(diag.clamp(min=1e-12)) - base  # (B,1,1)

        # Mix posterior over x0=k. K is small (e.g., 7), so a loop is fine and stable.
        stacked = []
        for k in range(self.K):
            row = base.unsqueeze(-1).expand(x_t.shape[0], x_t.shape[1], x_t.shape[2], self.K).clone()
            row[..., k] = row[..., k] + bump.expand(x_t.shape[0], x_t.shape[1], x_t.shape[2])
            post_logits = log_Qt + row
            post_logp = post_logits - torch.logsumexp(post_logits, dim=-1, keepdim=True)  # (B,H,W,K)
            stacked.append(torch.log(p_x0[..., k].clamp(min=1e-12)).unsqueeze(-1) + post_logp)
        return torch.logsumexp(torch.stack(stacked, dim=0), dim=0)  # (B,H,W,K)

    def save_config(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def from_config(cls, path: str | Path, device: str | torch.device = "cpu") -> "D3PMScheduler":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = D3PMConfig(**data)
        return cls(
            num_classes=int(cfg.num_classes),
            num_timesteps=int(cfg.num_timesteps),
            beta_start=float(cfg.beta_start),
            beta_end=float(cfg.beta_end),
            device=device,
        )
