"""Tests for the Semantic Associative Memory Substrate."""

import pytest
import torch

from brahman_os.memory.sams import SAMSEmbeddingSubstrate, TridoshaLoss


def test_sams_shapes_and_finite_outputs() -> None:
    """A SAMS pass should preserve sequence shape and contract to one embedding."""
    torch.manual_seed(7)
    batch_size, sequence_length, embedding_dim = 2, 5, 8
    model = SAMSEmbeddingSubstrate(embedding_dim=embedding_dim, tau=2.0)
    hidden_states = torch.randn(batch_size, sequence_length, embedding_dim)

    output = model(hidden_states)

    assert output.psi.shape == (batch_size, embedding_dim)
    assert output.reconstruction.shape == hidden_states.shape
    assert output.gamma.shape == hidden_states.shape
    assert output.krama.shape == (sequence_length, sequence_length)
    assert all(torch.isfinite(tensor).all() for tensor in output)


def test_reconstruction_loss_is_scalar_and_finite() -> None:
    """Reconstruction MSE should be a finite scalar."""
    torch.manual_seed(11)
    model = SAMSEmbeddingSubstrate(embedding_dim=4)
    hidden_states = torch.randn(3, 6, 4)

    output = model(hidden_states)
    loss = model.reconstruction_loss(output.reconstruction, hidden_states)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_krama_matrix_matches_relative_distance_formula() -> None:
    """Krama should encode exponentially decaying relative distance."""
    model = SAMSEmbeddingSubstrate(embedding_dim=3, tau=0.5)

    krama = model.krama_matrix(4)

    assert krama.shape == (4, 4)
    assert torch.allclose(torch.diag(krama), torch.ones(4))
    assert torch.allclose(krama, krama.T)
    assert torch.isclose(krama[0, 2], torch.exp(torch.tensor(-2.0)))


def test_tridosha_loss_returns_finite_scalar_components() -> None:
    """Tridosha loss should report finite entropy, reconstruction, and anchor terms."""
    torch.manual_seed(13)
    model = SAMSEmbeddingSubstrate(embedding_dim=6)
    hidden_states = torch.randn(2, 4, 6)
    output = model(hidden_states)
    anchor = torch.zeros_like(output.psi)

    losses = TridoshaLoss()(
        gamma=output.gamma,
        reconstruction=output.reconstruction,
        target=hidden_states,
        psi=output.psi,
        anchor=anchor,
    )

    assert all(loss.ndim == 0 for loss in losses)
    assert all(torch.isfinite(loss) for loss in losses)
    assert torch.isclose(losses.total, losses.vata + losses.pitta + losses.kapha)


def test_sams_is_deterministic_with_torch_seed() -> None:
    """Identical seeds, parameters, and inputs should produce identical outputs."""
    torch.manual_seed(23)
    first_model = SAMSEmbeddingSubstrate(embedding_dim=5, tau=1.5)
    hidden_states = torch.randn(2, 3, 5)
    first_output = first_model(hidden_states)

    torch.manual_seed(23)
    second_model = SAMSEmbeddingSubstrate(embedding_dim=5, tau=1.5)
    second_hidden_states = torch.randn(2, 3, 5)
    second_output = second_model(second_hidden_states)

    assert torch.equal(hidden_states, second_hidden_states)
    assert all(
        torch.equal(first_tensor, second_tensor)
        for first_tensor, second_tensor in zip(first_output, second_output, strict=True)
    )


@pytest.mark.parametrize(
    ("embedding_dim", "tau", "message"),
    [
        (0, 1.0, "embedding_dim must be positive"),
        (4, 0.0, "tau must be positive"),
    ],
)
def test_sams_rejects_invalid_configuration(
    embedding_dim: int,
    tau: float,
    message: str,
) -> None:
    """Invalid model dimensions and decay values should fail explicitly."""
    with pytest.raises(ValueError, match=message):
        SAMSEmbeddingSubstrate(embedding_dim=embedding_dim, tau=tau)


@pytest.mark.parametrize(
    ("hidden_states", "error_type", "message"),
    [
        (
            torch.randn(2, 4),
            ValueError,
            "hidden_states must have shape",
        ),
        (
            torch.randn(2, 0, 4),
            ValueError,
            "at least one sequence element",
        ),
        (
            torch.randn(2, 3, 5),
            ValueError,
            "dimension must equal embedding_dim",
        ),
        (
            torch.ones(2, 3, 4, dtype=torch.int64),
            TypeError,
            "floating-point dtype",
        ),
    ],
)
def test_sams_rejects_invalid_hidden_states(
    hidden_states: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    """Malformed hidden-state tensors should fail before tensor operations."""
    model = SAMSEmbeddingSubstrate(embedding_dim=4)

    with pytest.raises(error_type, match=message):
        model(hidden_states)


def test_losses_reject_incompatible_shapes() -> None:
    """Reconstruction and Tridosha losses should reject mismatched tensors."""
    model = SAMSEmbeddingSubstrate(embedding_dim=4)

    with pytest.raises(ValueError, match="identical shapes"):
        model.reconstruction_loss(torch.zeros(1, 2, 4), torch.zeros(1, 3, 4))

    with pytest.raises(ValueError, match="psi and anchor"):
        TridoshaLoss()(
            gamma=torch.full((1, 2, 4), 0.5),
            reconstruction=torch.zeros(1, 2, 4),
            target=torch.zeros(1, 2, 4),
            psi=torch.zeros(1, 4),
            anchor=torch.zeros(2, 4),
        )
