"""Tensor-based Semantic Associative Memory Substrate."""

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SAMSOutput(NamedTuple):
    """Outputs produced by a SAMS embedding and reconstruction pass."""

    psi: Tensor
    reconstruction: Tensor
    gamma: Tensor
    krama: Tensor


class TridoshaLossOutput(NamedTuple):
    """Explainable components of the Tridosha training objective."""

    total: Tensor
    vata: Tensor
    pitta: Tensor
    kapha: Tensor


class SAMSEmbeddingSubstrate(nn.Module):
    """Contract a sequence into a memory embedding and reconstruct its context."""

    def __init__(self, embedding_dim: int, tau: float = 1.0) -> None:
        """Initialize SAMS projections for a fixed embedding dimension."""
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if tau <= 0.0:
            raise ValueError("tau must be positive")

        self.embedding_dim = embedding_dim
        self.tau = float(tau)
        self.gate_projection = nn.Linear(embedding_dim, embedding_dim)
        self.sequence_projection = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.Linear(embedding_dim * 2, embedding_dim)

    def krama_matrix(
        self,
        sequence_length: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Build the relative-distance matrix for a sequence."""
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        positions = torch.arange(sequence_length, device=device, dtype=dtype or torch.float32)
        distances = torch.abs(positions[:, None] - positions[None, :])
        scale = max(self.tau, 1.0)
        return torch.exp(-distances / scale)

    def forward(self, hidden_states: Tensor) -> SAMSOutput:
        """Encode and reconstruct hidden states shaped ``[batch, sequence, dimension]``."""
        self._validate_hidden_states(hidden_states)

        gamma = torch.sigmoid(self.gate_projection(hidden_states))
        contracted = torch.sum(self.sequence_projection(hidden_states) * gamma, dim=1)
        psi = self.layer_norm(contracted)

        sequence_length = hidden_states.shape[1]
        krama = self.krama_matrix(
            sequence_length,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        context = torch.einsum("ij,bjd->bid", krama, hidden_states)
        psi_expanded = psi.unsqueeze(1).expand(-1, sequence_length, -1)
        reconstruction = self.decoder(torch.cat((psi_expanded, context), dim=-1))

        return SAMSOutput(
            psi=psi,
            reconstruction=reconstruction,
            gamma=gamma,
            krama=krama,
        )

    @staticmethod
    def reconstruction_loss(reconstruction: Tensor, target: Tensor) -> Tensor:
        """Return scalar mean-squared reconstruction error."""
        if reconstruction.shape != target.shape:
            raise ValueError("reconstruction and target must have identical shapes")
        return F.mse_loss(reconstruction, target)

    def _validate_hidden_states(self, hidden_states: Tensor) -> None:
        """Validate the SAMS input tensor contract."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, dimension]")
        if hidden_states.shape[1] <= 0:
            raise ValueError("hidden_states must contain at least one sequence element")
        if hidden_states.shape[2] != self.embedding_dim:
            raise ValueError(
                f"hidden_states dimension must equal embedding_dim ({self.embedding_dim})"
            )
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must use a floating-point dtype")


class TridoshaLoss(nn.Module):
    """Combine entropy, reconstruction, and anchor-stability losses."""

    def __init__(
        self,
        *,
        vata_weight: float = 1.0,
        pitta_weight: float = 1.0,
        kapha_weight: float = 1.0,
        epsilon: float = 1e-7,
    ) -> None:
        """Initialize non-negative weights for each Tridosha component."""
        super().__init__()
        weights = (vata_weight, pitta_weight, kapha_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Tridosha weights must be non-negative")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.vata_weight = float(vata_weight)
        self.pitta_weight = float(pitta_weight)
        self.kapha_weight = float(kapha_weight)
        self.epsilon = float(epsilon)

    def forward(
        self,
        *,
        gamma: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        psi: Tensor,
        anchor: Tensor,
    ) -> TridoshaLossOutput:
        """Return the total loss and named component losses."""
        if gamma.numel() == 0:
            raise ValueError("gamma must not be empty")
        if reconstruction.shape != target.shape:
            raise ValueError("reconstruction and target must have identical shapes")
        if psi.shape != anchor.shape:
            raise ValueError("psi and anchor must have identical shapes")

        bounded_gamma = gamma.clamp(self.epsilon, 1.0 - self.epsilon)
        vata = -(
            bounded_gamma * bounded_gamma.log()
            + (1.0 - bounded_gamma) * (1.0 - bounded_gamma).log()
        ).mean()
        pitta = F.mse_loss(reconstruction, target)
        kapha = F.mse_loss(psi, anchor)
        total = (
            self.vata_weight * vata
            + self.pitta_weight * pitta
            + self.kapha_weight * kapha
        )
        return TridoshaLossOutput(total=total, vata=vata, pitta=pitta, kapha=kapha)


SAMS = SAMSEmbeddingSubstrate
