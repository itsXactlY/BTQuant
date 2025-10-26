import pytest

torch = pytest.importorskip("torch")

from neural_trading_system.models.architecture import (
    PositionalEncoding,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerBlock,
    MarketRegimeVAE,
    RegimeChangeDetector,
    ProfitTakingModule,
    StopLossModule,
    LetWinnerRunModule,
    NeuralTradingModel,
    create_model,
)


def test_positional_encoding_adds_signal():
    encoding = PositionalEncoding(d_model=8, max_len=4)
    x = torch.zeros(2, 4, 8)
    encoded = encoding(x)

    assert encoded.shape == x.shape
    # Positional encoding should inject non-zero signal into the zeros tensor
    assert not torch.allclose(encoded, torch.zeros_like(encoded))


def test_multi_head_attention_shapes():
    attention = MultiHeadSelfAttention(d_model=16, num_heads=4)
    x = torch.randn(3, 5, 16)

    output, attn_weights = attention(x)

    assert output.shape == x.shape
    assert attn_weights.shape == (3, 4, 5, 5)
    assert torch.isfinite(output).all()
    assert torch.isfinite(attn_weights).all()


def test_feed_forward_preserves_shape():
    feed_forward = FeedForward(d_model=12, d_ff=24)
    x = torch.randn(2, 7, 12)

    output = feed_forward(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_transformer_block_outputs_attention():
    block = TransformerBlock(d_model=20, num_heads=4, d_ff=40)
    x = torch.randn(2, 6, 20)

    output, attn_weights = block(x)

    assert output.shape == x.shape
    assert attn_weights.shape == (2, 4, 6, 6)
    assert torch.isfinite(output).all()


def test_market_regime_vae_handles_numerical_instability():
    vae = MarketRegimeVAE(input_dim=16, latent_dim=6)
    x = torch.randn(4, 16)
    x[0, 0] = float("nan")
    x[1, 1] = float("inf")
    x[2, 2] = float("-inf")

    recon, mu, logvar, z = vae(x)

    for tensor in (recon, mu, logvar, z):
        assert torch.isfinite(tensor).all()

    assert recon.shape == x.shape
    assert mu.shape == (4, 6)
    assert logvar.shape == (4, 6)
    assert z.shape == (4, 6)
    assert logvar.min().item() >= -10 - 1e-5
    assert logvar.max().item() <= 10 + 1e-5


def test_regime_change_detector_outputs_probabilities():
    detector = RegimeChangeDetector(d_model=18, latent_dim=6)
    current_repr = torch.randn(3, 18)
    current_regime = torch.randn(3, 6)
    historical_regime_mean = torch.randn(3, 6)

    outputs = detector(current_repr, current_regime, historical_regime_mean)

    expected_keys = {
        "regime_change_score",
        "stability",
        "vol_change",
        "volume_anomaly",
        "trend_break",
        "liquidity_shift",
    }
    assert set(outputs) == expected_keys

    for value in outputs.values():
        assert value.shape == (3, 1)
        assert torch.all((0.0 <= value) & (value <= 1.0))


def _assert_probability_dict(output, expected_keys):
    assert set(output) == set(expected_keys)
    for key in expected_keys:
        tensor = output[key]
        assert tensor.shape[-1] == 1
        assert torch.isfinite(tensor).all()
        assert torch.all((0.0 <= tensor) & (tensor <= 1.0))


def test_profit_taking_module_outputs_probabilities():
    module = ProfitTakingModule(d_model=14, latent_dim=5)
    batch = 4
    current_repr = torch.randn(batch, 14)
    regime_z = torch.randn(batch, 5)
    pnl = torch.rand(batch, 1)
    time_in_position = torch.rand(batch, 1)
    expected_return = torch.rand(batch, 1)

    outputs = module(current_repr, regime_z, pnl, time_in_position, expected_return)

    expected_keys = {
        "take_profit_prob",
        "momentum_fade",
        "resistance_near",
        "profit_optimal",
        "extension_risk",
    }
    _assert_probability_dict(outputs, expected_keys)


def test_stop_loss_module_outputs_probabilities():
    module = StopLossModule(d_model=14, latent_dim=5)
    batch = 3
    current_repr = torch.randn(batch, 14)
    regime_z = torch.randn(batch, 5)
    pnl = -torch.rand(batch, 1)
    regime_change = torch.rand(batch, 1)
    time_in_position = torch.rand(batch, 1)

    outputs = module(current_repr, regime_z, pnl, regime_change, time_in_position)

    expected_keys = {
        "stop_loss_prob",
        "pattern_failure",
        "acceleration_down",
        "support_break",
        "regime_hostile",
    }
    _assert_probability_dict(outputs, expected_keys)


def test_let_winner_run_module_outputs_probabilities():
    module = LetWinnerRunModule(d_model=14, latent_dim=5)
    batch = 2
    current_repr = torch.randn(batch, 14)
    regime_z = torch.randn(batch, 5)
    pnl = torch.rand(batch, 1)
    momentum = torch.rand(batch, 1)
    stability = torch.rand(batch, 1)

    outputs = module(current_repr, regime_z, pnl, momentum, stability)

    expected_keys = {
        "hold_score",
        "trend_strength",
        "momentum_accel",
        "breakout_confirm",
        "regime_favorable",
    }
    _assert_probability_dict(outputs, expected_keys)


def _create_small_model():
    return NeuralTradingModel(
        feature_dim=10,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        dropout=0.1,
        latent_dim=8,
        seq_len=5,
    )


def test_neural_trading_model_without_position_context():
    model = _create_small_model()
    x = torch.randn(2, 5, 10)

    outputs = model(x)

    expected_top_level = {
        "entry_logits",
        "entry_prob",
        "expected_return",
        "volatility_forecast",
        "position_size",
        "regime_mu",
        "regime_logvar",
        "regime_z",
        "vae_recon",
        "attention_weights",
        "sequence_repr",
        "regime_change",
        "exit_signals",
        "unified_exit_prob",
    }
    assert set(outputs) == expected_top_level

    exit_signals = outputs["exit_signals"]
    assert exit_signals == {
        "profit_taking": None,
        "stop_loss": None,
        "let_winner_run": None,
    }

    unified_exit_prob = outputs["unified_exit_prob"]
    assert unified_exit_prob.shape == (2, 1)
    assert torch.allclose(unified_exit_prob, torch.zeros_like(unified_exit_prob))


def test_neural_trading_model_with_profit_position_context():
    model = _create_small_model()
    x = torch.randn(2, 5, 10)
    position_context = {
        "unrealized_pnl": torch.full((2, 1), 0.25),
        "time_in_position": torch.full((2, 1), 0.5),
    }

    outputs = model(x, position_context=position_context)

    exit_signals = outputs["exit_signals"]
    assert set(exit_signals) >= {"profit_taking", "let_winner_run", "should_exit_profit"}
    assert exit_signals["profit_taking"] is not None
    assert exit_signals["let_winner_run"] is not None
    should_exit = exit_signals["should_exit_profit"]
    assert should_exit.shape == (2, 1)
    assert torch.all((0.0 <= should_exit) & (should_exit <= 1.0))
    assert torch.allclose(outputs["unified_exit_prob"], should_exit)


def test_neural_trading_model_with_loss_position_context():
    model = _create_small_model()
    x = torch.randn(2, 5, 10)
    position_context = {
        "unrealized_pnl": torch.full((2, 1), -0.3),
        "time_in_position": torch.full((2, 1), 0.4),
    }

    outputs = model(x, position_context=position_context)

    exit_signals = outputs["exit_signals"]
    assert set(exit_signals) >= {"stop_loss", "should_exit_loss"}
    assert exit_signals["stop_loss"] is not None
    should_exit = exit_signals["should_exit_loss"]
    assert should_exit.shape == (2, 1)
    assert torch.all((0.0 <= should_exit) & (should_exit <= 1.0))
    assert torch.allclose(outputs["unified_exit_prob"], should_exit)


def test_unified_exit_prob_returns_zeros_without_signals():
    model = _create_small_model()
    stability = torch.full((3, 1), 0.5)
    regime_info = {"stability": stability}
    exit_signals = {"profit_taking": None, "stop_loss": None, "let_winner_run": None}

    unified = model._compute_unified_exit(exit_signals, regime_info)

    assert unified.shape == stability.shape
    assert torch.allclose(unified, torch.zeros_like(unified))


def test_create_model_applies_config_overrides():
    config = {"d_model": 64, "num_heads": 8, "num_layers": 3, "latent_dim": 12, "seq_len": 50}
    model = create_model(feature_dim=15, config=config)

    assert isinstance(model, NeuralTradingModel)
    assert model.d_model == 64
    assert len(model.transformer_blocks) == 3
    assert model.latent_dim == 12
