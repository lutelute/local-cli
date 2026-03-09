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


class TestAntigravityConfigKeys(unittest.TestCase):
    """Tests for new antigravity config keys: plan_dir, knowledge_dir, skills_dir, default_mode."""

    def setUp(self) -> None:
        """Save and clear relevant env vars to isolate tests."""
        self._saved_env: dict[str, str] = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                self._saved_env[env_var] = os.environ.pop(env_var)

    def tearDown(self) -> None:
        """Restore saved env vars."""
        os.environ.update(self._saved_env)

    def test_new_keys_in_config_defaults(self) -> None:
        """New antigravity keys are present in CONFIG_DEFAULTS."""
        self.assertIn("plan_dir", CONFIG_DEFAULTS)
        self.assertIn("knowledge_dir", CONFIG_DEFAULTS)
        self.assertIn("skills_dir", CONFIG_DEFAULTS)
        self.assertIn("default_mode", CONFIG_DEFAULTS)

    def test_default_plan_dir(self) -> None:
        """Default plan_dir is .agents/plans."""
        cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
        self.assertEqual(cfg.plan_dir, ".agents/plans")

    def test_default_knowledge_dir(self) -> None:
        """Default knowledge_dir is .agents/knowledge."""
        cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
        self.assertEqual(cfg.knowledge_dir, ".agents/knowledge")

    def test_default_skills_dir(self) -> None:
        """Default skills_dir is .agents/skills."""
        cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
        self.assertEqual(cfg.skills_dir, ".agents/skills")

    def test_default_mode(self) -> None:
        """Default default_mode is 'agent'."""
        cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
        self.assertEqual(cfg.default_mode, "agent")

    def test_config_defaults_values(self) -> None:
        """CONFIG_DEFAULTS contains expected values for new keys."""
        self.assertEqual(CONFIG_DEFAULTS["plan_dir"], ".agents/plans")
        self.assertEqual(CONFIG_DEFAULTS["knowledge_dir"], ".agents/knowledge")
        self.assertEqual(CONFIG_DEFAULTS["skills_dir"], ".agents/skills")
        self.assertEqual(CONFIG_DEFAULTS["default_mode"], "agent")

    def test_config_file_overrides_plan_dir(self) -> None:
        """Config file overrides plan_dir default."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("plan_dir=custom/plans\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertEqual(cfg.plan_dir, "custom/plans")
        finally:
            os.unlink(path)

    def test_config_file_overrides_knowledge_dir(self) -> None:
        """Config file overrides knowledge_dir default."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("knowledge_dir=/var/knowledge\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertEqual(cfg.knowledge_dir, "/var/knowledge")
        finally:
            os.unlink(path)

    def test_config_file_overrides_skills_dir(self) -> None:
        """Config file overrides skills_dir default."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("skills_dir=my_skills\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertEqual(cfg.skills_dir, "my_skills")
        finally:
            os.unlink(path)

    def test_config_file_overrides_default_mode(self) -> None:
        """Config file overrides default_mode default."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("default_mode=ideate\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertEqual(cfg.default_mode, "ideate")
        finally:
            os.unlink(path)

    def test_config_file_overrides_all_new_keys(self) -> None:
        """Config file can override all new keys at once."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("plan_dir=custom/plans\n")
            f.write("knowledge_dir=custom/knowledge\n")
            f.write("skills_dir=custom/skills\n")
            f.write("default_mode=plan\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertEqual(cfg.plan_dir, "custom/plans")
            self.assertEqual(cfg.knowledge_dir, "custom/knowledge")
            self.assertEqual(cfg.skills_dir, "custom/skills")
            self.assertEqual(cfg.default_mode, "plan")
        finally:
            os.unlink(path)

    def test_cli_args_override_plan_dir(self) -> None:
        """CLI args override plan_dir from config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("plan_dir=file-plans\n")
            path = f.name

        try:
            cli = SimpleNamespace(plan_dir="cli-plans")
            cfg = Config(cli_args=cli, config_file=path)
            self.assertEqual(cfg.plan_dir, "cli-plans")
        finally:
            os.unlink(path)

    def test_cli_args_override_knowledge_dir(self) -> None:
        """CLI args override knowledge_dir from config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("knowledge_dir=file-knowledge\n")
            path = f.name

        try:
            cli = SimpleNamespace(knowledge_dir="cli-knowledge")
            cfg = Config(cli_args=cli, config_file=path)
            self.assertEqual(cfg.knowledge_dir, "cli-knowledge")
        finally:
            os.unlink(path)

    def test_cli_args_override_default_mode(self) -> None:
        """CLI args override default_mode from config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("default_mode=agent\n")
            path = f.name

        try:
            cli = SimpleNamespace(default_mode="ideate")
            cfg = Config(cli_args=cli, config_file=path)
            self.assertEqual(cfg.default_mode, "ideate")
        finally:
            os.unlink(path)

    def test_cli_none_does_not_override_new_keys(self) -> None:
        """CLI args with None values do not override new key defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("plan_dir=file-plans\n")
            path = f.name

        try:
            cli = SimpleNamespace(plan_dir=None, knowledge_dir=None)
            cfg = Config(cli_args=cli, config_file=path)
            # plan_dir: file value preserved (CLI was None).
            self.assertEqual(cfg.plan_dir, "file-plans")
            # knowledge_dir: default preserved (CLI was None, no file value).
            self.assertEqual(cfg.knowledge_dir, ".agents/knowledge")
        finally:
            os.unlink(path)

    def test_new_keys_are_str_type(self) -> None:
        """All new config keys are stored as str type."""
        cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
        self.assertIsInstance(cfg.plan_dir, str)
        self.assertIsInstance(cfg.knowledge_dir, str)
        self.assertIsInstance(cfg.skills_dir, str)
        self.assertIsInstance(cfg.default_mode, str)

    def test_new_keys_coexist_with_existing_defaults(self) -> None:
        """New keys do not interfere with existing config defaults."""
        cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
        # Existing defaults still work.
        self.assertEqual(cfg.model, CONFIG_DEFAULTS["model"])
        self.assertFalse(cfg.debug)
        self.assertFalse(cfg.auto_approve)
        # New defaults also work.
        self.assertEqual(cfg.plan_dir, ".agents/plans")
        self.assertEqual(cfg.default_mode, "agent")

    def test_mixed_old_and_new_keys_in_config_file(self) -> None:
        """Config file with both old and new keys works correctly."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("model=custom-model\n")
            f.write("debug=true\n")
            f.write("plan_dir=my/plans\n")
            f.write("default_mode=plan\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertEqual(cfg.model, "custom-model")
            self.assertTrue(cfg.debug)
            self.assertEqual(cfg.plan_dir, "my/plans")
            self.assertEqual(cfg.default_mode, "plan")
            # Unset new keys fall back to defaults.
            self.assertEqual(cfg.knowledge_dir, ".agents/knowledge")
            self.assertEqual(cfg.skills_dir, ".agents/skills")
        finally:
            os.unlink(path)



class TestConfigNumCtx(unittest.TestCase):
    """Tests for num_ctx config key."""

    def _save_env(self) -> dict[str, str]:
        saved: dict[str, str] = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved[env_var] = os.environ.pop(env_var)
        return saved

    def test_config_num_ctx_default(self) -> None:
        """Default num_ctx is 8192 when nothing is configured."""
        saved_env = self._save_env()
        try:
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertEqual(cfg.num_ctx, 8192)
        finally:
            os.environ.update(saved_env)

    def test_config_num_ctx_from_env(self) -> None:
        """LOCAL_CLI_NUM_CTX env var sets num_ctx."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_NUM_CTX"] = "16384"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertEqual(cfg.num_ctx, 16384)
        finally:
            os.environ.pop("LOCAL_CLI_NUM_CTX", None)
            os.environ.update(saved_env)

    def test_config_num_ctx_from_cli_arg(self) -> None:
        """CLI arg num_ctx overrides env and default."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_NUM_CTX"] = "16384"
            cli = SimpleNamespace(num_ctx=32768)
            cfg = Config(
                cli_args=cli,
                config_file="/tmp/nonexistent_local_cli_config",
            )
            self.assertEqual(cfg.num_ctx, 32768)
        finally:
            os.environ.pop("LOCAL_CLI_NUM_CTX", None)
            os.environ.update(saved_env)

    def test_config_num_ctx_from_config_file(self) -> None:
        """Config file num_ctx overrides default."""
        saved_env = self._save_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("num_ctx=4096\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertEqual(cfg.num_ctx, 4096)
        finally:
            os.unlink(path)
            os.environ.update(saved_env)

    def test_config_num_ctx_is_int(self) -> None:
        """num_ctx is always an int, even when loaded from string sources."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_NUM_CTX"] = "8192"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertIsInstance(cfg.num_ctx, int)
        finally:
            os.environ.pop("LOCAL_CLI_NUM_CTX", None)
            os.environ.update(saved_env)


class TestConfigTemperature(unittest.TestCase):
    """Tests for temperature config key."""

    def _save_env(self) -> dict[str, str]:
        saved: dict[str, str] = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved[env_var] = os.environ.pop(env_var)
        return saved

    def test_config_temperature_none_when_unset(self) -> None:
        """Temperature is None when not configured."""
        saved_env = self._save_env()
        try:
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertIsNone(cfg.temperature)
        finally:
            os.environ.update(saved_env)

    def test_config_temperature_float_parsing(self) -> None:
        """Temperature is parsed as float from env var string."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_TEMPERATURE"] = "0.6"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertAlmostEqual(cfg.temperature, 0.6)
            self.assertIsInstance(cfg.temperature, float)
        finally:
            os.environ.pop("LOCAL_CLI_TEMPERATURE", None)
            os.environ.update(saved_env)

    def test_config_temperature_from_config_file(self) -> None:
        """Temperature parsed from config file as float."""
        saved_env = self._save_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("temperature=0.8\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertAlmostEqual(cfg.temperature, 0.8)
        finally:
            os.unlink(path)
            os.environ.update(saved_env)

    def test_config_temperature_from_cli_arg(self) -> None:
        """CLI arg temperature overrides env."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_TEMPERATURE"] = "0.6"
            cli = SimpleNamespace(temperature=0.3)
            cfg = Config(
                cli_args=cli,
                config_file="/tmp/nonexistent_local_cli_config",
            )
            self.assertAlmostEqual(cfg.temperature, 0.3)
        finally:
            os.environ.pop("LOCAL_CLI_TEMPERATURE", None)
            os.environ.update(saved_env)

    def test_config_temperature_zero(self) -> None:
        """Temperature=0.0 is a valid explicit value (not treated as None)."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_TEMPERATURE"] = "0.0"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertAlmostEqual(cfg.temperature, 0.0)
            self.assertIsNotNone(cfg.temperature)
        finally:
            os.environ.pop("LOCAL_CLI_TEMPERATURE", None)
            os.environ.update(saved_env)


class TestConfigThinkMode(unittest.TestCase):
    """Tests for think_mode config key."""

    def _save_env(self) -> dict[str, str]:
        saved: dict[str, str] = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved[env_var] = os.environ.pop(env_var)
        return saved

    def test_config_think_mode_default_false(self) -> None:
        """think_mode defaults to False."""
        saved_env = self._save_env()
        try:
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertFalse(cfg.think_mode)
        finally:
            os.environ.update(saved_env)

    def test_config_think_mode_bool(self) -> None:
        """think_mode parsed as bool from env var."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_THINK_MODE"] = "true"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertTrue(cfg.think_mode)
            self.assertIsInstance(cfg.think_mode, bool)
        finally:
            os.environ.pop("LOCAL_CLI_THINK_MODE", None)
            os.environ.update(saved_env)

    def test_config_think_mode_from_config_file(self) -> None:
        """think_mode parsed from config file as bool."""
        saved_env = self._save_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("think_mode=1\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertTrue(cfg.think_mode)
        finally:
            os.unlink(path)
            os.environ.update(saved_env)

    def test_config_think_mode_from_cli_arg(self) -> None:
        """CLI arg think_mode overrides lower layers."""
        saved_env = self._save_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("think_mode=false\n")
            path = f.name

        try:
            cli = SimpleNamespace(think_mode=True)
            cfg = Config(cli_args=cli, config_file=path)
            self.assertTrue(cfg.think_mode)
        finally:
            os.unlink(path)
            os.environ.update(saved_env)


class TestConfigTopPTopK(unittest.TestCase):
    """Tests for top_p and top_k config keys."""

    def _save_env(self) -> dict[str, str]:
        saved: dict[str, str] = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved[env_var] = os.environ.pop(env_var)
        return saved

    def test_config_top_p_none_when_unset(self) -> None:
        """top_p is None when not configured."""
        saved_env = self._save_env()
        try:
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertIsNone(cfg.top_p)
        finally:
            os.environ.update(saved_env)

    def test_config_top_p_float_parsing(self) -> None:
        """top_p is parsed as float from env var."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_TOP_P"] = "0.95"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertAlmostEqual(cfg.top_p, 0.95)
            self.assertIsInstance(cfg.top_p, float)
        finally:
            os.environ.pop("LOCAL_CLI_TOP_P", None)
            os.environ.update(saved_env)

    def test_config_top_k_none_when_unset(self) -> None:
        """top_k is None when not configured."""
        saved_env = self._save_env()
        try:
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertIsNone(cfg.top_k)
        finally:
            os.environ.update(saved_env)

    def test_config_top_k_int_parsing(self) -> None:
        """top_k is parsed as int from env var."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_TOP_K"] = "20"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertEqual(cfg.top_k, 20)
            self.assertIsInstance(cfg.top_k, int)
        finally:
            os.environ.pop("LOCAL_CLI_TOP_K", None)
            os.environ.update(saved_env)


class TestConfigKeepAlive(unittest.TestCase):
    """Tests for keep_alive config key."""

    def _save_env(self) -> dict[str, str]:
        saved: dict[str, str] = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved[env_var] = os.environ.pop(env_var)
        return saved

    def test_config_keep_alive_none_when_unset(self) -> None:
        """keep_alive is None when not configured."""
        saved_env = self._save_env()
        try:
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertIsNone(cfg.keep_alive)
        finally:
            os.environ.update(saved_env)

    def test_config_keep_alive_from_env(self) -> None:
        """keep_alive set from env var."""
        saved_env = self._save_env()
        try:
            os.environ["LOCAL_CLI_KEEP_ALIVE"] = "5m"
            cfg = Config(config_file="/tmp/nonexistent_local_cli_config")
            self.assertEqual(cfg.keep_alive, "5m")
        finally:
            os.environ.pop("LOCAL_CLI_KEEP_ALIVE", None)
            os.environ.update(saved_env)

    def test_config_keep_alive_from_config_file(self) -> None:
        """keep_alive set from config file."""
        saved_env = self._save_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("keep_alive=10m\n")
            path = f.name

        try:
            cfg = Config(config_file=path)
            self.assertEqual(cfg.keep_alive, "10m")
        finally:
            os.unlink(path)
            os.environ.update(saved_env)


class TestConfigNewKeysPriorityChain(unittest.TestCase):
    """Integration test: priority chain for new inference config keys."""

    def _save_env(self) -> dict[str, str]:
        saved: dict[str, str] = {}
        for env_var in ENV_VAR_MAP:
            if env_var in os.environ:
                saved[env_var] = os.environ.pop(env_var)
        return saved

    def test_config_priority_chain_num_ctx(self) -> None:
        """Verify full priority chain for num_ctx: CLI > env > file > defaults."""
        saved_env = self._save_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("num_ctx=4096\n")
            f.write("temperature=0.5\n")
            f.write("think_mode=true\n")
            path = f.name

        try:
            # File overrides default.
            cfg_file_only = Config(config_file=path)
            self.assertEqual(cfg_file_only.num_ctx, 4096)

            # Env overrides file.
            os.environ["LOCAL_CLI_NUM_CTX"] = "16384"
            cfg_env = Config(config_file=path)
            self.assertEqual(cfg_env.num_ctx, 16384)

            # CLI overrides env.
            cli = SimpleNamespace(num_ctx=32768)
            cfg_cli = Config(cli_args=cli, config_file=path)
            self.assertEqual(cfg_cli.num_ctx, 32768)
        finally:
            os.environ.pop("LOCAL_CLI_NUM_CTX", None)
            os.unlink(path)
            os.environ.update(saved_env)

    def test_config_priority_chain_temperature(self) -> None:
        """Verify full priority chain for temperature."""
        saved_env = self._save_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("temperature=0.5\n")
            path = f.name

        try:
            # Default: None (no file, no env, no CLI).
            cfg_default = Config(
                config_file="/tmp/nonexistent_local_cli_config"
            )
            self.assertIsNone(cfg_default.temperature)

            # File overrides default.
            cfg_file = Config(config_file=path)
            self.assertAlmostEqual(cfg_file.temperature, 0.5)

            # Env overrides file.
            os.environ["LOCAL_CLI_TEMPERATURE"] = "0.8"
            cfg_env = Config(config_file=path)
            self.assertAlmostEqual(cfg_env.temperature, 0.8)

            # CLI overrides env.
            cli = SimpleNamespace(temperature=0.2)
            cfg_cli = Config(cli_args=cli, config_file=path)
            self.assertAlmostEqual(cfg_cli.temperature, 0.2)
        finally:
            os.environ.pop("LOCAL_CLI_TEMPERATURE", None)
            os.unlink(path)
            os.environ.update(saved_env)

    def test_config_priority_chain_think_mode(self) -> None:
        """Verify full priority chain for think_mode."""
        saved_env = self._save_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("think_mode=true\n")
            path = f.name

        try:
            # Default: False.
            cfg_default = Config(
                config_file="/tmp/nonexistent_local_cli_config"
            )
            self.assertFalse(cfg_default.think_mode)

            # File sets to True.
            cfg_file = Config(config_file=path)
            self.assertTrue(cfg_file.think_mode)

            # Env overrides file.
            os.environ["LOCAL_CLI_THINK_MODE"] = "false"
            cfg_env = Config(config_file=path)
            self.assertFalse(cfg_env.think_mode)

            # CLI overrides env.
            cli = SimpleNamespace(think_mode=True)
            cfg_cli = Config(cli_args=cli, config_file=path)
            self.assertTrue(cfg_cli.think_mode)
        finally:
            os.environ.pop("LOCAL_CLI_THINK_MODE", None)
            os.unlink(path)
            os.environ.update(saved_env)

    def test_config_all_new_keys_from_mixed_sources(self) -> None:
        """Different new keys from different layers coexist correctly."""
        saved_env = self._save_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        ) as f:
            f.write("num_ctx=4096\n")
            f.write("temperature=0.5\n")
            path = f.name

        try:
            # num_ctx from env, temperature from file, think_mode from CLI.
            os.environ["LOCAL_CLI_NUM_CTX"] = "16384"
            cli = SimpleNamespace(think_mode=True)
            cfg = Config(cli_args=cli, config_file=path)

            self.assertEqual(cfg.num_ctx, 16384)  # env overrides file
            self.assertAlmostEqual(cfg.temperature, 0.5)  # file (no env/CLI)
            self.assertTrue(cfg.think_mode)  # CLI
            self.assertIsNone(cfg.top_p)  # default (not set anywhere)
            self.assertIsNone(cfg.top_k)  # default (not set anywhere)
        finally:
            os.environ.pop("LOCAL_CLI_NUM_CTX", None)
            os.unlink(path)
            os.environ.update(saved_env)



if __name__ == "__main__":
    unittest.main()
