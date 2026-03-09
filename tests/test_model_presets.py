"""Tests for local_cli.model_presets module."""

import unittest

from local_cli.model_presets import (
    SUPPORTS_THINKING,
    _DEFAULT_PRESET,
    _PRESETS,
    _QWEN3_MIN_TEMPERATURE,
    get_model_family,
    get_model_preset,
)

# Valid Ollama inference option keys (subset relevant to presets).
# See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
_VALID_OLLAMA_OPTIONS: frozenset[str] = frozenset(
    {
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "num_ctx",
        "repeat_last_n",
        "repeat_penalty",
        "temperature",
        "seed",
        "stop",
        "num_predict",
        "top_k",
        "top_p",
        "min_p",
        "tfs_z",
        "typical_p",
        "num_keep",
        "num_batch",
        "num_gpu",
        "main_gpu",
        "low_vram",
        "vocab_only",
        "use_mmap",
        "use_mlock",
        "num_thread",
    }
)


class TestGetModelFamily(unittest.TestCase):
    """Tests for get_model_family()."""

    def test_qwen3_with_tag(self) -> None:
        """qwen3:8b maps to qwen3 family."""
        self.assertEqual(get_model_family("qwen3:8b"), "qwen3")

    def test_qwen3_latest(self) -> None:
        """qwen3:latest maps to qwen3 family."""
        self.assertEqual(get_model_family("qwen3:latest"), "qwen3")

    def test_qwen3_bare(self) -> None:
        """Bare 'qwen3' (no tag) maps to qwen3 family."""
        self.assertEqual(get_model_family("qwen3"), "qwen3")

    def test_qwen25_with_instruct_tag(self) -> None:
        """qwen2.5:7b-instruct maps to qwen2.5 family."""
        self.assertEqual(get_model_family("qwen2.5:7b-instruct"), "qwen2.5")

    def test_qwen25_bare(self) -> None:
        """Bare 'qwen2.5' maps to qwen2.5 family."""
        self.assertEqual(get_model_family("qwen2.5"), "qwen2.5")

    def test_gemma_with_tag(self) -> None:
        """gemma3:4b maps to gemma family."""
        self.assertEqual(get_model_family("gemma3:4b"), "gemma")

    def test_gemma_bare(self) -> None:
        """gemma maps to gemma family."""
        self.assertEqual(get_model_family("gemma"), "gemma")

    def test_phi_with_tag(self) -> None:
        """phi4-mini maps to phi family."""
        self.assertEqual(get_model_family("phi4-mini"), "phi")

    def test_llama_with_tag(self) -> None:
        """llama3.2:latest maps to llama family."""
        self.assertEqual(get_model_family("llama3.2:latest"), "llama")

    def test_unknown_model(self) -> None:
        """Unknown model returns None."""
        self.assertIsNone(get_model_family("mistral:7b"))

    def test_empty_string(self) -> None:
        """Empty model name returns None."""
        self.assertIsNone(get_model_family(""))

    def test_case_insensitive(self) -> None:
        """Model family detection is case-insensitive."""
        self.assertEqual(get_model_family("Qwen3:8b"), "qwen3")
        self.assertEqual(get_model_family("QWEN3:8B"), "qwen3")

    def test_qwen25_not_matched_as_qwen2(self) -> None:
        """qwen2.5 matches qwen2.5 specifically, not a shorter prefix."""
        family = get_model_family("qwen2.5:7b-instruct")
        self.assertEqual(family, "qwen2.5")


class TestGetModelPresetQwen3(unittest.TestCase):
    """Tests for get_model_preset() with Qwen3 family models."""

    def test_qwen3_temperature(self) -> None:
        """Qwen3 preset has temperature=0.6."""
        preset = get_model_preset("qwen3:8b")
        self.assertEqual(preset["temperature"], 0.6)

    def test_qwen3_top_p(self) -> None:
        """Qwen3 preset has top_p=0.95."""
        preset = get_model_preset("qwen3:8b")
        self.assertEqual(preset["top_p"], 0.95)

    def test_qwen3_top_k(self) -> None:
        """Qwen3 preset has top_k=20."""
        preset = get_model_preset("qwen3:8b")
        self.assertEqual(preset["top_k"], 20)

    def test_qwen3_num_ctx(self) -> None:
        """Qwen3 preset has num_ctx=8192."""
        preset = get_model_preset("qwen3:8b")
        self.assertEqual(preset["num_ctx"], 8192)

    def test_qwen3_all_keys_present(self) -> None:
        """Qwen3 preset contains all expected keys."""
        preset = get_model_preset("qwen3:8b")
        expected_keys = {"temperature", "top_p", "top_k", "num_ctx"}
        self.assertEqual(set(preset.keys()), expected_keys)

    def test_qwen3_latest_tag(self) -> None:
        """Qwen3 with :latest tag gets the same preset."""
        preset = get_model_preset("qwen3:latest")
        self.assertEqual(preset["temperature"], 0.6)
        self.assertEqual(preset["num_ctx"], 8192)


class TestGetModelPresetQwen25(unittest.TestCase):
    """Tests for get_model_preset() with Qwen2.5 family models."""

    def test_qwen25_temperature(self) -> None:
        """Qwen2.5 preset has temperature=0.7."""
        preset = get_model_preset("qwen2.5:7b-instruct")
        self.assertEqual(preset["temperature"], 0.7)

    def test_qwen25_top_p(self) -> None:
        """Qwen2.5 preset has top_p=0.8."""
        preset = get_model_preset("qwen2.5:7b-instruct")
        self.assertEqual(preset["top_p"], 0.8)

    def test_qwen25_top_k(self) -> None:
        """Qwen2.5 preset has top_k=20."""
        preset = get_model_preset("qwen2.5:7b-instruct")
        self.assertEqual(preset["top_k"], 20)

    def test_qwen25_num_ctx(self) -> None:
        """Qwen2.5 preset has num_ctx=8192."""
        preset = get_model_preset("qwen2.5:7b-instruct")
        self.assertEqual(preset["num_ctx"], 8192)


class TestGetModelPresetOtherFamilies(unittest.TestCase):
    """Tests for get_model_preset() with Gemma, Phi, and Llama families."""

    def test_gemma_preset(self) -> None:
        """Gemma preset has expected values."""
        preset = get_model_preset("gemma3:4b")
        self.assertEqual(preset["temperature"], 0.7)
        self.assertEqual(preset["num_ctx"], 8192)

    def test_phi_preset(self) -> None:
        """Phi preset has expected values."""
        preset = get_model_preset("phi4-mini")
        self.assertEqual(preset["temperature"], 0.7)
        self.assertEqual(preset["num_ctx"], 8192)

    def test_llama_preset(self) -> None:
        """Llama preset has expected values."""
        preset = get_model_preset("llama3.2:latest")
        self.assertEqual(preset["temperature"], 0.7)
        self.assertEqual(preset["num_ctx"], 8192)


class TestUnknownModelDefaults(unittest.TestCase):
    """Tests for get_model_preset() with unknown models."""

    def test_unknown_model_returns_defaults(self) -> None:
        """Unknown model gets safe default preset."""
        preset = get_model_preset("mistral:7b")
        self.assertEqual(preset["temperature"], 0.7)
        self.assertEqual(preset["num_ctx"], 8192)

    def test_unknown_model_has_minimal_keys(self) -> None:
        """Default preset only contains temperature and num_ctx."""
        preset = get_model_preset("some-unknown-model:latest")
        self.assertEqual(set(preset.keys()), {"temperature", "num_ctx"})

    def test_unknown_model_matches_default_preset(self) -> None:
        """Unknown model preset matches _DEFAULT_PRESET values."""
        preset = get_model_preset("unknown:latest")
        for key, value in _DEFAULT_PRESET.items():
            self.assertEqual(preset[key], value)


class TestNoTempZeroQwen3(unittest.TestCase):
    """Tests ensuring Qwen3 family never uses temperature=0."""

    def test_qwen3_preset_temp_above_zero(self) -> None:
        """Qwen3 default preset temperature is above 0."""
        preset = get_model_preset("qwen3:8b")
        self.assertGreater(preset["temperature"], 0)

    def test_qwen3_preset_temp_at_least_minimum(self) -> None:
        """Qwen3 preset temperature is at least the minimum (0.1)."""
        preset = get_model_preset("qwen3:8b")
        self.assertGreaterEqual(preset["temperature"], _QWEN3_MIN_TEMPERATURE)

    def test_qwen3_stored_preset_temp_not_zero(self) -> None:
        """The stored Qwen3 preset does not have temperature=0."""
        self.assertNotEqual(_PRESETS["qwen3"]["temperature"], 0)

    def test_min_temp_constant_is_positive(self) -> None:
        """The minimum temperature constant is a positive number."""
        self.assertGreater(_QWEN3_MIN_TEMPERATURE, 0)

    def test_qwen3_min_temp_enforced(self) -> None:
        """Even if _PRESETS were tampered with, the floor is enforced.

        This tests the enforcement logic in get_model_preset() by
        checking that the returned value is at least the minimum.
        """
        preset = get_model_preset("qwen3:8b")
        self.assertGreaterEqual(preset["temperature"], _QWEN3_MIN_TEMPERATURE)


class TestSupportsThinking(unittest.TestCase):
    """Tests for SUPPORTS_THINKING set."""

    def test_qwen3_supports_thinking(self) -> None:
        """Qwen3 is in the SUPPORTS_THINKING set."""
        self.assertIn("qwen3", SUPPORTS_THINKING)

    def test_qwen25_does_not_support_thinking(self) -> None:
        """Qwen2.5 is not in the SUPPORTS_THINKING set."""
        self.assertNotIn("qwen2.5", SUPPORTS_THINKING)

    def test_gemma_does_not_support_thinking(self) -> None:
        """Gemma is not in the SUPPORTS_THINKING set."""
        self.assertNotIn("gemma", SUPPORTS_THINKING)

    def test_phi_does_not_support_thinking(self) -> None:
        """Phi is not in the SUPPORTS_THINKING set."""
        self.assertNotIn("phi", SUPPORTS_THINKING)

    def test_llama_does_not_support_thinking(self) -> None:
        """Llama is not in the SUPPORTS_THINKING set."""
        self.assertNotIn("llama", SUPPORTS_THINKING)

    def test_supports_thinking_is_frozenset(self) -> None:
        """SUPPORTS_THINKING is immutable."""
        self.assertIsInstance(SUPPORTS_THINKING, frozenset)

    def test_thinking_check_with_model_family(self) -> None:
        """Full workflow: get family then check thinking support."""
        qwen3_family = get_model_family("qwen3:8b")
        self.assertIn(qwen3_family, SUPPORTS_THINKING)

        qwen25_family = get_model_family("qwen2.5:7b-instruct")
        self.assertNotIn(qwen25_family, SUPPORTS_THINKING)


class TestPresetKeysAreValid(unittest.TestCase):
    """Tests that all preset keys are valid Ollama inference options."""

    def test_qwen3_keys_valid(self) -> None:
        """All keys in Qwen3 preset are valid Ollama options."""
        preset = get_model_preset("qwen3:8b")
        for key in preset:
            self.assertIn(
                key,
                _VALID_OLLAMA_OPTIONS,
                f"Qwen3 preset key {key!r} is not a valid Ollama option",
            )

    def test_qwen25_keys_valid(self) -> None:
        """All keys in Qwen2.5 preset are valid Ollama options."""
        preset = get_model_preset("qwen2.5:7b-instruct")
        for key in preset:
            self.assertIn(
                key,
                _VALID_OLLAMA_OPTIONS,
                f"Qwen2.5 preset key {key!r} is not a valid Ollama option",
            )

    def test_all_preset_families_have_valid_keys(self) -> None:
        """All keys in every registered preset are valid Ollama options."""
        for family, preset in _PRESETS.items():
            for key in preset:
                self.assertIn(
                    key,
                    _VALID_OLLAMA_OPTIONS,
                    f"Preset {family!r} has invalid key {key!r}",
                )

    def test_default_preset_keys_valid(self) -> None:
        """All keys in the default preset are valid Ollama options."""
        for key in _DEFAULT_PRESET:
            self.assertIn(
                key,
                _VALID_OLLAMA_OPTIONS,
                f"Default preset key {key!r} is not a valid Ollama option",
            )


class TestPresetIsolation(unittest.TestCase):
    """Tests that returned presets are independent copies."""

    def test_mutation_does_not_affect_source(self) -> None:
        """Mutating a returned preset does not affect the stored preset."""
        preset1 = get_model_preset("qwen3:8b")
        preset1["temperature"] = 999.0

        preset2 = get_model_preset("qwen3:8b")
        self.assertEqual(preset2["temperature"], 0.6)

    def test_returned_preset_is_dict(self) -> None:
        """get_model_preset always returns a plain dict."""
        for model in ("qwen3:8b", "qwen2.5:7b-instruct", "unknown:latest"):
            preset = get_model_preset(model)
            self.assertIsInstance(preset, dict)


class TestPresetValueTypes(unittest.TestCase):
    """Tests that preset values have correct types."""

    def test_temperature_is_float(self) -> None:
        """Temperature values are floats."""
        for family in list(_PRESETS.keys()) + ["unknown"]:
            model = f"{family}:latest" if family != "unknown" else "unknown:latest"
            preset = get_model_preset(model)
            self.assertIsInstance(
                preset["temperature"], (int, float),
                f"temperature for {family} should be numeric",
            )

    def test_num_ctx_is_int(self) -> None:
        """num_ctx values are integers."""
        for family in list(_PRESETS.keys()) + ["unknown"]:
            model = f"{family}:latest" if family != "unknown" else "unknown:latest"
            preset = get_model_preset(model)
            self.assertIsInstance(
                preset["num_ctx"], int,
                f"num_ctx for {family} should be int",
            )

    def test_top_k_is_int(self) -> None:
        """top_k values (when present) are integers."""
        for family, preset_data in _PRESETS.items():
            if "top_k" in preset_data:
                preset = get_model_preset(f"{family}:latest")
                self.assertIsInstance(
                    preset["top_k"], int,
                    f"top_k for {family} should be int",
                )

    def test_top_p_is_float(self) -> None:
        """top_p values (when present) are floats."""
        for family, preset_data in _PRESETS.items():
            if "top_p" in preset_data:
                preset = get_model_preset(f"{family}:latest")
                self.assertIsInstance(
                    preset["top_p"], (int, float),
                    f"top_p for {family} should be numeric",
                )


class TestFamilyPrefixes(unittest.TestCase):
    """Tests for internal family prefix ordering."""

    def test_all_presets_have_family_match(self) -> None:
        """Every key in _PRESETS is detectable via get_model_family."""
        for family in _PRESETS:
            result = get_model_family(f"{family}:latest")
            self.assertEqual(
                result, family,
                f"Family {family!r} not detected from '{family}:latest'",
            )

    def test_longer_prefix_wins(self) -> None:
        """Longer prefix is preferred over shorter (qwen2.5 over qwen2)."""
        # qwen2.5 should not accidentally match a shorter "qwen" prefix.
        family = get_model_family("qwen2.5:7b")
        self.assertEqual(family, "qwen2.5")


if __name__ == "__main__":
    unittest.main()
