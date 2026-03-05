"""Tests for local_cli.config module."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from local_cli.config import (
    CONFIG_DEFAULTS,
    ENV_VAR_MAP,
    Config,
    _parse_bool,
    load_config_file,
)


class TestLoadConfigFile(unittest.TestCase):
    """Tests for load_config_file()."""

    def test_basic_key_value(self) -> None:
        """Parse a simple key=value config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("model=qwen3:8b\n")
            f.write("debug=true\n")
            path = f.name

        try:
            result = load_config_file(path)
            self.assertEqual(result["model"], "qwen3:8b")
            self.assertEqual(result["debug"], "true")
        finally:
            os.unlink(path)

    def test_comments_and_blank_lines(self) -> None:
        """Skip comments and blank lines."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("# This is a comment\n")
            f.write("\n")
            f.write("model=qwen3:8b\n")
            f.write("  # another comment\n")
            f.write("\n")
            f.write("debug=1\n")
            path = f.name

        try:
            result = load_config_file(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result["model"], "qwen3:8b")
            self.assertEqual(result["debug"], "1")
        finally:
            os.unlink(path)

    def test_value_with_equals_sign(self) -> None:
        """Only split on first '=' so values can contain '='."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("ollama_host=http://localhost:11434\n")
            f.write("some_key=value=with=equals\n")
            path = f.name

        try:
            result = load_config_file(path)
            self.assertEqual(result["ollama_host"], "http://localhost:11434")
            self.assertEqual(result["some_key"], "value=with=equals")
        finally:
            os.unlink(path)

    def test_whitespace_stripping(self) -> None:
        """Whitespace around keys and values is stripped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("  model  =  qwen3:8b  \n")
            f.write("  debug =true\n")
            path = f.name

        try:
            result = load_config_file(path)
            self.assertEqual(result["model"], "qwen3:8b")
            self.assertEqual(result["debug"], "true")
        finally:
            os.unlink(path)

    def test_missing_file(self) -> None:
        """Return empty dict for a non-existent file."""
        result = load_config_file("/tmp/does_not_exist_local_cli_test.conf")
        self.assertEqual(result, {})

    def test_empty_file(self) -> None:
        """Return empty dict for an empty config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            path = f.name

        try:
            result = load_config_file(path)
            self.assertEqual(result, {})
        finally:
            os.unlink(path)

    def test_lines_without_equals(self) -> None:
        """Lines without '=' are silently skipped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("no_equals_here\n")
            f.write("model=qwen3:8b\n")
            path = f.name

        try:
            result = load_config_file(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result["model"], "qwen3:8b")
        finally:
            os.unlink(path)

    def test_symlink_rejected(self) -> None:
        """Symlinked config files are rejected."""
        tmpdir = tempfile.mkdtemp()
        real_path = os.path.join(tmpdir, "real_config")
        link_path = os.path.join(tmpdir, "link_config")

        try:
            with open(real_path, "w") as f:
                f.write("model=evil_model\n")
            os.symlink(real_path, link_path)

            result = load_config_file(link_path)
            self.assertEqual(result, {})
        finally:
            os.unlink(link_path)
            os.unlink(real_path)
            os.rmdir(tmpdir)

    def test_oversized_config_rejected(self) -> None:
        """Config files exceeding 10KB are rejected."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            # Write more than 10KB of data.
            f.write("key=value\n" * 2000)
            path = f.name

        try:
            # Verify the file is actually >10KB.
            self.assertGreater(os.path.getsize(path), 10 * 1024)
            result = load_config_file(path)
            self.assertEqual(result, {})
        finally:
            os.unlink(path)

    def test_shell_injection_characters_treated_as_literal(self) -> None:
        """Shell injection characters are parsed as literal values, not executed."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("model=$(rm -rf /)\n")
            f.write("debug=`whoami`\n")
            f.write("sidecar_model=; echo pwned\n")
            f.write("ollama_host=http://localhost:11434 && curl evil.com\n")
            path = f.name

        try:
            result = load_config_file(path)
            # Values are parsed literally -- no shell expansion.
            self.assertEqual(result["model"], "$(rm -rf /)")
            self.assertEqual(result["debug"], "`whoami`")
            self.assertEqual(result["sidecar_model"], "; echo pwned")
            self.assertEqual(
                result["ollama_host"],
                "http://localhost:11434 && curl evil.com",
            )
        finally:
            os.unlink(path)

    def test_empty_key_skipped(self) -> None:
        """Lines with empty keys (e.g., '=value') are skipped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("=no_key\n")
            f.write("  =also_no_key\n")
            f.write("model=qwen3:8b\n")
            path = f.name

        try:
            result = load_config_file(path)
            self.assertEqual(len(result), 1)
            self.assertIn("model", result)
        finally:
            os.unlink(path)

    def test_directory_rejected(self) -> None:
        """Directories are not accepted as config files."""
        tmpdir = tempfile.mkdtemp()
        try:
            result = load_config_file(tmpdir)
            self.assertEqual(result, {})
        finally:
            os.rmdir(tmpdir)


class TestParseBool(unittest.TestCase):
    """Tests for _parse_bool()."""

    def test_bool_true(self) -> None:
        self.assertTrue(_parse_bool(True))

    def test_bool_false(self) -> None:
        self.assertFalse(_parse_bool(False))

    def test_string_true_variants(self) -> None:
        for val in ("1", "true", "True", "TRUE", "yes", "Yes", "YES"):
            self.assertTrue(_parse_bool(val), f"Expected True for {val!r}")

    def test_string_false_variants(self) -> None:
        for val in ("0", "false", "False", "no", "No", "", "other"):
            self.assertFalse(_parse_bool(val), f"Expected False for {val!r}")

    def test_int_truthy(self) -> None:
        self.assertTrue(_parse_bool(1))

    def test_int_falsy(self) -> None:
        self.assertFalse(_parse_bool(0))


class TestConfigDefaults(unittest.TestCase):
    """Tests for Config defaults fallback."""

    def test_defaults_without_file_or_env(self) -> None:
        """Config with no file, no env, no CLI args uses defaults."""
        # Use a non-existent config file to avoid loading any real config.
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        try:
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertEqual(cfg.model, CONFIG_DEFAULTS["model"])
            self.assertEqual(cfg.sidecar_model, CONFIG_DEFAULTS["sidecar_model"])
            self.assertEqual(
                cfg.ollama_host,
                str(CONFIG_DEFAULTS["ollama_host"]).rstrip("/"),
            )
            self.assertFalse(cfg.debug)
            self.assertFalse(cfg.auto_approve)
        finally:
            os.environ.update(saved_env)


class TestConfigFileLayer(unittest.TestCase):
    """Tests for config file layer in Config."""

    def test_config_file_overrides_defaults(self) -> None:
        """Values in config file override defaults."""
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("model=custom-model:latest\n")
            f.write("debug=true\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertEqual(cfg.model, "custom-model:latest")
            self.assertTrue(cfg.debug)
        finally:
            os.unlink(path)
            os.environ.update(saved_env)

    def test_unknown_keys_in_config_file_ignored(self) -> None:
        """Keys not in CONFIG_DEFAULTS are ignored."""
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("unknown_key=should_be_ignored\n")
            f.write("model=qwen3:8b\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertFalse(hasattr(cfg, "unknown_key"))
            self.assertEqual(cfg.model, "qwen3:8b")
        finally:
            os.unlink(path)
            os.environ.update(saved_env)


class TestConfigEnvVarLayer(unittest.TestCase):
    """Tests for environment variable layer in Config."""

    def test_env_var_overrides_config_file(self) -> None:
        """Environment variables override config file values."""
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("model=file-model\n")
            path = f.name

        try:
            os.environ["LOCAL_CLI_MODEL"] = "env-model"
            cfg = Config(config_file=path)
            self.assertEqual(cfg.model, "env-model")
        finally:
            os.environ.pop("LOCAL_CLI_MODEL", None)
            os.unlink(path)
            os.environ.update(saved_env)

    def test_env_var_overrides_defaults(self) -> None:
        """Environment variables override defaults."""
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        try:
            os.environ["LOCAL_CLI_DEBUG"] = "1"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertTrue(cfg.debug)
        finally:
            os.environ.pop("LOCAL_CLI_DEBUG", None)
            os.environ.update(saved_env)

    def test_ollama_host_env_override(self) -> None:
        """OLLAMA_HOST env var overrides config."""
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        try:
            os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11435"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertEqual(cfg.ollama_host, "http://127.0.0.1:11435")
        finally:
            os.environ.pop("OLLAMA_HOST", None)
            os.environ.update(saved_env)


class TestConfigCLIArgLayer(unittest.TestCase):
    """Tests for CLI argument layer in Config."""

    def test_cli_args_override_env_and_file(self) -> None:
        """CLI args have highest priority."""
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("model=file-model\n")
            path = f.name

        try:
            os.environ["LOCAL_CLI_MODEL"] = "env-model"
            cli = SimpleNamespace(model="cli-model", debug=None)
            cfg = Config(cli_args=cli, config_file=path)
            self.assertEqual(cfg.model, "cli-model")
        finally:
            os.environ.pop("LOCAL_CLI_MODEL", None)
            os.unlink(path)
            os.environ.update(saved_env)

    def test_cli_none_values_do_not_override(self) -> None:
        """CLI args with None values do not override lower layers."""
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("model=file-model\n")
            path = f.name

        try:
            cli = SimpleNamespace(model=None, debug=None)
            cfg = Config(cli_args=cli, config_file=path)
            self.assertEqual(cfg.model, "file-model")
        finally:
            os.unlink(path)
            os.environ.update(saved_env)

    def test_cli_debug_flag(self) -> None:
        """CLI debug flag as boolean overrides correctly."""
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        try:
            cli = SimpleNamespace(model=None, debug=True)
            cfg = Config(
                cli_args=cli,
                config_file="/tmp/nonexistent_local_cli_config",
            )
            self.assertTrue(cfg.debug)
        finally:
            os.environ.update(saved_env)


class TestConfigPriorityIntegration(unittest.TestCase):
    """Integration test: full priority chain defaults < file < env < CLI."""

    def test_full_priority_chain(self) -> None:
        """Verify the complete priority: CLI > env > file > defaults."""
        saved_env = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved_env[env_var] = os.environ.pop(env_var)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("model=file-model\n")
            f.write("sidecar_model=file-sidecar\n")
            f.write("debug=false\n")
            path = f.name

        try:
            # Env overrides file for sidecar_model.
            os.environ["LOCAL_CLI_SIDECAR_MODEL"] = "env-sidecar"
            # CLI overrides everything for model.
            cli = SimpleNamespace(model="cli-model", debug=None, sidecar_model=None)
            cfg = Config(cli_args=cli, config_file=path)

            # model: CLI arg wins.
            self.assertEqual(cfg.model, "cli-model")
            # sidecar_model: env wins over file.
            self.assertEqual(cfg.sidecar_model, "env-sidecar")
            # debug: file value (no env or CLI override).
            self.assertFalse(cfg.debug)
        finally:
            os.environ.pop("LOCAL_CLI_SIDECAR_MODEL", None)
            os.unlink(path)
            os.environ.update(saved_env)


if __name__ == "__main__":
    unittest.main()
