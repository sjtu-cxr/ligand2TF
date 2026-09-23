import pytest
import torch

from src.unified_dstar_model import UnifiedDStar


@pytest.fixture
def inputs():
    torch.manual_seed(7)
    return {
        "molformer": torch.randn(3, 6),
        "ecfp": torch.randn(3, 9),
        "ion_features": torch.randn(3, 4),
        "ion_mask": torch.tensor([False, True, False]),
        "esm": torch.randn(5, 7),
    }


@pytest.mark.parametrize("architecture", ["A", "B"])
def test_shape_and_unit_norm_contract(architecture, inputs):
    model = UnifiedDStar(
        architecture,
        molformer_dim=6,
        ecfp_dim=9,
        ion_dim=4,
        esm_dim=7,
        hidden_dim=11,
        contrast_dim=5,
        dropout=0.0,
    )

    ligand_z = model.encode_ligand(
        inputs["molformer"],
        inputs["ecfp"],
        inputs["ion_features"],
        inputs["ion_mask"],
    )
    protein_z = model.encode_protein(inputs["esm"])

    assert ligand_z.shape == (3, 5)
    assert protein_z.shape == (5, 5)
    torch.testing.assert_close(ligand_z.norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(protein_z.norm(dim=-1), torch.ones(5))


def test_score_has_pairwise_shape_and_learned_temperature(inputs):
    model = UnifiedDStar(
        "A",
        molformer_dim=6,
        ecfp_dim=9,
        ion_dim=4,
        esm_dim=7,
        hidden_dim=11,
        contrast_dim=5,
        dropout=0.0,
        temperature_init=0.25,
    )
    ligand_z = model.encode_ligand(
        inputs["molformer"],
        inputs["ecfp"],
        inputs["ion_features"],
        inputs["ion_mask"],
    )
    protein_z = model.encode_protein(inputs["esm"])

    scores = model.score(ligand_z, protein_z)

    assert scores.shape == (3, 5)
    torch.testing.assert_close(scores, 4.0 * ligand_z @ protein_z.T)


def test_extreme_logit_scale_has_finite_forward_and_backward():
    model = UnifiedDStar(
        "A",
        molformer_dim=3,
        ecfp_dim=4,
        ion_dim=2,
        esm_dim=5,
        hidden_dim=7,
        contrast_dim=3,
        dropout=0.0,
    )
    model.logit_scale.data.fill_(1.0e30)
    ligand_z = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    protein_z = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)

    scores = model.score(ligand_z, protein_z)
    scores.sum().backward()

    assert torch.isfinite(scores).all()
    assert model.logit_scale.grad is not None
    assert torch.isfinite(model.logit_scale.grad)


@pytest.mark.parametrize("temperature_init", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_temperature_init_fails(temperature_init):
    with pytest.raises(ValueError, match="temperature_init must be finite and positive"):
        UnifiedDStar("A", temperature_init=temperature_init)


@pytest.mark.parametrize("residual_gate_init", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_residual_gate_init_fails(residual_gate_init):
    with pytest.raises(ValueError, match="residual_gate_init must be finite"):
        UnifiedDStar("B", residual_gate_init=residual_gate_init)


def test_architecture_b_has_no_geometry_parameters():
    model = UnifiedDStar("B")

    assert all("geom" not in name.lower() for name, _ in model.named_parameters())


def test_unknown_architecture_fails():
    with pytest.raises(ValueError, match="architecture must be 'A' or 'B'"):
        UnifiedDStar("C")


def test_architecture_a_uses_ecfp_fallback_and_ion_override():
    torch.manual_seed(3)
    model = UnifiedDStar(
        "A",
        molformer_dim=3,
        ecfp_dim=4,
        ion_dim=2,
        esm_dim=5,
        hidden_dim=7,
        contrast_dim=3,
        dropout=0.0,
    )
    model.eval()
    molformer = torch.tensor(
        [[0.0, 0.0, 0.0], [float("nan"), 1.0, 2.0], [1.0, 2.0, 3.0]]
    )
    ecfp = torch.randn(3, 4)
    ion_features = torch.randn(3, 2)
    ion_mask = torch.tensor([False, False, True])

    ligand_z = model.encode_ligand(molformer, ecfp, ion_features, ion_mask)
    expected_ecfp = torch.nn.functional.normalize(model.ecfp_projection(ecfp[:2]), dim=-1)
    expected_ion = torch.nn.functional.normalize(model.ion_projection(ion_features[2:]), dim=-1)

    torch.testing.assert_close(ligand_z[:2], expected_ecfp)
    torch.testing.assert_close(ligand_z[2:], expected_ion)
    assert torch.isfinite(ligand_z).all()


def test_architecture_b_matches_exact_gated_formula_in_eval_mode():
    torch.manual_seed(11)
    model = UnifiedDStar(
        "B",
        molformer_dim=3,
        ecfp_dim=4,
        ion_dim=2,
        esm_dim=5,
        hidden_dim=7,
        contrast_dim=3,
        dropout=0.0,
        residual_gate_init=-2.0,
    )
    model.eval()
    molformer = torch.tensor(
        [[1.0, 2.0, 3.0], [float("nan"), float("inf"), float("-inf")]]
    )
    ecfp = torch.randn(2, 4)
    ion_features = torch.randn(2, 2)
    ion_mask = torch.zeros(2, dtype=torch.bool)
    esm = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0],
            [float("nan"), float("inf"), float("-inf"), 0.0, 1.0],
        ]
    )

    ligand_z = model.encode_ligand(molformer, ecfp, ion_features, ion_mask)
    protein_z = model.encode_protein(esm)

    neutral_molformer = torch.nan_to_num(
        molformer, nan=0.0, posinf=0.0, neginf=0.0
    )
    ligand_expected = torch.nn.functional.normalize(
        model.molformer_projection(neutral_molformer)
        + torch.sigmoid(model.ligand_residual_gate) * model.ecfp_projection(ecfp),
        dim=-1,
    )
    neutral_esm = torch.nan_to_num(esm, nan=0.0, posinf=0.0, neginf=0.0)
    protein_expected = torch.nn.functional.normalize(
        model.protein_projection(
            neutral_esm
            + torch.sigmoid(model.protein_residual_gate)
            * model.protein_residual_adapter(neutral_esm)
        ),
        dim=-1,
    )
    torch.testing.assert_close(ligand_z, ligand_expected)
    torch.testing.assert_close(protein_z, protein_expected)
    assert torch.isfinite(ligand_z).all()
    assert torch.isfinite(protein_z).all()


def test_architecture_b_ion_path_overrides_regular_ligand_path():
    torch.manual_seed(13)
    model = UnifiedDStar(
        "B",
        molformer_dim=3,
        ecfp_dim=4,
        ion_dim=2,
        esm_dim=5,
        hidden_dim=7,
        contrast_dim=3,
        dropout=0.0,
    )
    model.eval()
    molformer = torch.tensor([[float("nan"), float("inf"), float("-inf")]])
    ecfp = torch.full((1, 4), float("nan"))
    ion_features = torch.tensor([[0.5, -1.5]])

    ligand_z = model.encode_ligand(
        molformer, ecfp, ion_features, torch.tensor([True])
    )
    expected = torch.nn.functional.normalize(
        model.ion_projection(ion_features), dim=-1
    )

    torch.testing.assert_close(ligand_z, expected)
    assert torch.isfinite(ligand_z).all()


def test_architecture_b_gates_exist_and_both_residual_paths_get_gradients(inputs):
    model = UnifiedDStar(
        "B",
        molformer_dim=6,
        ecfp_dim=9,
        ion_dim=4,
        esm_dim=7,
        hidden_dim=11,
        contrast_dim=5,
        dropout=0.0,
        residual_gate_init=-3.0,
    )
    assert model.ligand_residual_gate.ndim == 0
    assert model.protein_residual_gate.ndim == 0
    torch.testing.assert_close(model.ligand_residual_gate.detach(), torch.tensor(-3.0))
    torch.testing.assert_close(model.protein_residual_gate.detach(), torch.tensor(-3.0))

    ligand_z = model.encode_ligand(
        inputs["molformer"],
        inputs["ecfp"],
        inputs["ion_features"],
        torch.zeros(3, dtype=torch.bool),
    )
    protein_z = model.encode_protein(inputs["esm"])
    model.score(ligand_z, protein_z).sum().backward()

    assert model.ligand_residual_gate.grad is not None
    assert torch.isfinite(model.ligand_residual_gate.grad)
    assert model.ligand_residual_gate.grad.abs().item() > 0
    assert model.protein_residual_gate.grad is not None
    assert torch.isfinite(model.protein_residual_gate.grad)
    assert model.protein_residual_gate.grad.abs().item() > 0
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum().item() > 0
        for parameter in model.ecfp_projection.parameters()
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum().item() > 0
        for parameter in model.protein_residual_adapter.parameters()
    )


@pytest.mark.parametrize("architecture", ["A", "B"])
def test_masked_nonfinite_placeholders_keep_outputs_and_parameter_gradients_finite(
    architecture,
):
    torch.manual_seed(17)
    model = UnifiedDStar(
        architecture,
        molformer_dim=3,
        ecfp_dim=4,
        ion_dim=2,
        esm_dim=5,
        hidden_dim=7,
        contrast_dim=3,
        dropout=0.0,
    )
    molformer = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [float("nan"), float("inf"), float("-inf")],
        ]
    )
    ecfp = torch.tensor(
        [
            [float("nan"), float("inf"), float("-inf"), float("nan")],
            [1.0, 0.0, 1.0, 0.0],
            [float("nan"), float("inf"), float("-inf"), float("nan")],
        ]
    )
    ion_features = torch.tensor(
        [
            [float("nan"), float("inf")],
            [float("-inf"), float("nan")],
            [0.25, -0.75],
        ]
    )
    ion_mask = torch.tensor([False, False, True])
    esm = torch.randn(2, 5)

    ligand_z = model.encode_ligand(molformer, ecfp, ion_features, ion_mask)
    protein_z = model.encode_protein(esm)
    loss = model.score(ligand_z, protein_z).sum()
    loss.backward()

    assert torch.isfinite(ligand_z).all()
    assert torch.isfinite(protein_z).all()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize(
    ("method", "arguments", "message"),
    [
        (
            "encode_ligand",
            (torch.randn(2, 6, 1), torch.randn(2, 9), torch.randn(2, 4), torch.zeros(2, dtype=torch.bool)),
            "molformer must have shape",
        ),
        (
            "encode_ligand",
            (torch.randn(2, 6), torch.randn(3, 9), torch.randn(2, 4), torch.zeros(2, dtype=torch.bool)),
            "ecfp must have shape",
        ),
        (
            "encode_ligand",
            (torch.randn(2, 6), torch.randn(2, 9), torch.randn(2, 5), torch.zeros(2, dtype=torch.bool)),
            "ion_features must have shape",
        ),
        (
            "encode_ligand",
            (torch.randn(2, 6), torch.randn(2, 9), torch.randn(2, 4), torch.zeros(2, 1, dtype=torch.bool)),
            "ion_mask must have shape",
        ),
        (
            "encode_ligand",
            (torch.randn(2, 6), torch.randn(2, 9), torch.randn(2, 4), torch.zeros(2)),
            "ion_mask must have dtype bool",
        ),
        ("encode_protein", (torch.randn(2, 8),), "esm must have shape"),
        ("score", (torch.randn(2, 4), torch.randn(3, 5)), "ligand_z must have shape"),
    ],
)
def test_invalid_input_shapes_fail(method, arguments, message):
    model = UnifiedDStar(
        "A",
        molformer_dim=6,
        ecfp_dim=9,
        ion_dim=4,
        esm_dim=7,
        hidden_dim=11,
        contrast_dim=5,
    )

    with pytest.raises(ValueError, match=message):
        getattr(model, method)(*arguments)
