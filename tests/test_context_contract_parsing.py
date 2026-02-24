from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
CANONICAL_CORE = BASE / "busy-38-ongoing"


def _install_minimal_core_stubs() -> None:
    if "core" in sys.modules and "core.orchestration.integration" in sys.modules:
        return

    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = []
    sys.modules.setdefault("core", core_pkg)

    orchestration_pkg = types.ModuleType("core.orchestration")
    orchestration_pkg.__path__ = []
    sys.modules.setdefault("core.orchestration", orchestration_pkg)

    class OrchestratorConfig:
        def __init__(self, *args, **kwargs):
            pass

    class Busy38Orchestrator:
        def __init__(self, *args, **kwargs):
            pass

    integration = types.ModuleType("core.orchestration.integration")
    integration.OrchestratorConfig = OrchestratorConfig
    integration.Busy38Orchestrator = Busy38Orchestrator
    sys.modules["core.orchestration.integration"] = integration

    plugins_pkg = types.ModuleType("core.plugins")
    plugins_pkg.__path__ = []
    sys.modules.setdefault("core.plugins", plugins_pkg)

    class PluginManager:
        pass

    plugins_mgr = types.ModuleType("core.plugins.manager")
    plugins_mgr.PluginManager = PluginManager
    sys.modules["core.plugins.manager"] = plugins_mgr

    security_pkg = types.ModuleType("core.security")
    security_pkg.__path__ = []
    sys.modules.setdefault("core.security", security_pkg)

    class HardwareAuthError(Exception):
        pass

    def require_hardware_auth(*_args, **_kwargs):
        return None

    def hardware_auth_is_configured(*_args, **_kwargs):
        return False

    def machine_fingerprint() -> str:
        return "test-fingerprint"

    hardware_auth = types.ModuleType("core.security.hardware_auth")
    hardware_auth.require_hardware_auth = require_hardware_auth
    hardware_auth.hardware_auth_is_configured = hardware_auth_is_configured
    hardware_auth.HardwareAuthError = HardwareAuthError
    hardware_auth.machine_fingerprint = machine_fingerprint
    sys.modules["core.security.hardware_auth"] = hardware_auth

    redaction_pkg = types.ModuleType("core.security.redaction")
    redaction_pkg.scrub_secrets = lambda text: text
    redaction_pkg.truncate = lambda text, max_len=512: (text if text is None else str(text)[:max_len])
    sys.modules["core.security.redaction"] = redaction_pkg

    session_pkg = types.ModuleType("core.session")

    class SessionStore:
        def __init__(self, *args, **kwargs):
            pass

    session_pkg.SessionStore = SessionStore
    sys.modules["core.session"] = session_pkg

    attachments_pkg = types.ModuleType("core.attachments")
    attachments_pkg.__path__ = []
    sys.modules.setdefault("core.attachments", attachments_pkg)

    intake_pkg = types.ModuleType("core.attachments.intake")
    intake_pkg.ATTACHMENT_DECISION_ACCEPT = "ACCEPT"
    intake_pkg.ATTACHMENT_DECISION_BLOCK = "BLOCK"
    intake_pkg.ATTACHMENT_DECISION_QUARANTINE = "QUARANTINE"
    intake_pkg.attachment_summary_line = lambda result, preview=None, descriptor=None: "attachment"
    sys.modules["core.attachments.intake"] = intake_pkg

    cognition_pkg = types.ModuleType("core.cognition")
    cognition_pkg.__path__ = []
    sys.modules.setdefault("core.cognition", cognition_pkg)

    cognition_intake_pkg = types.ModuleType("core.cognition.attachment_intake")
    cognition_intake_pkg.ATTACHMENT_DECISION_ACCEPT = "ACCEPT"
    cognition_intake_pkg.ATTACHMENT_DECISION_BLOCK = "BLOCK"
    cognition_intake_pkg.ATTACHMENT_DECISION_QUARANTINE = "QUARANTINE"
    cognition_intake_pkg.assess_text_ingress = lambda *args, **kwargs: ("ACCEPT", None, None, {})
    cognition_intake_pkg.extract_attachment_text_preview = lambda *args, **kwargs: ""
    cognition_intake_pkg.make_intake_decision = lambda *args, **kwargs: ("ACCEPT", None, None, {})
    cognition_intake_pkg.sanitize_attachment_for_transcript = lambda *args, **kwargs: ""
    sys.modules["core.cognition.attachment_intake"] = cognition_intake_pkg

    # Ensure local toolkit package is preferred over any canonical package copies.
    for path in (str(ROOT), str(BASE), str(CANONICAL_CORE)):
        if path in sys.path:
            sys.path.remove(path)
            sys.path.insert(0, path)

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *args, **kwargs: {}
    yaml_stub.dump = lambda *args, **kwargs: ""
    sys.modules["yaml"] = yaml_stub


def _load_discord_bot_module():
    _install_minimal_core_stubs()
    package_name = "toolkit_test_context_contract"
    toolkit_pkg = types.ModuleType(package_name)
    toolkit_pkg.__path__ = [str(ROOT / "toolkit")]
    sys.modules[package_name] = toolkit_pkg
    path = ROOT / "toolkit" / "discord_bot.py"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.discord_bot",
        str(path),
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{package_name}.discord_bot"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_install_minimal_core_stubs()
discord_bot = _load_discord_bot_module()
Busy38DiscordBot = discord_bot.Busy38DiscordBot


def _bot():
    # Avoid __init__ side effects (Discord token / network setup) by creating only the object shell.
    return Busy38DiscordBot.__new__(Busy38DiscordBot)


def test_extract_context_metadata_prefers_modern_contract_fields():
    bot = _bot()
    payload = {
        "context_contract": {"context_schema_version": 2, "context_source": "session", "context_budget_tokens": 8000},
        "context_payload": {"role": "assistant", "context_budget_tokens": 8000, "budget_usage_tokens": 150},
    }

    context = bot._extract_context_metadata(payload)

    assert context["context_schema_version"] == 2
    assert context["context_source"] == "session"
    assert context["context_budget_tokens"] == 8000
    assert context["budget_usage_tokens"] == 150
    assert context["context_contract_compat_mode"] == "modern"
    assert context["context_contract"] == payload["context_contract"]
    assert context["context_payload"] == payload["context_payload"]


def test_extract_context_metadata_handles_legacy_nested_contract():
    bot = _bot()
    payload = {
        "context_payload": {
            "context_contract": {
                "context_schema_version": 1,
                "context_source": "resume",
                "context_budget_tokens": 1200,
                "budget_usage_tokens": 12,
                "agent_profile_version": 3,
            }
        }
    }

    context = bot._extract_context_metadata(payload)

    assert context["context_schema_version"] == 1
    assert context["context_source"] == "resume"
    assert context["context_budget_tokens"] == 1200
    assert context["budget_usage_tokens"] == 12
    assert context["agent_profile_version"] == 3
    assert context["context_contract_compat_mode"] == "legacy"
    assert context["context_payload"] == payload["context_payload"]


def test_extract_context_metadata_preserves_legacy_compat_mode_override():
    bot = _bot()
    payload = {
        "context_contract_compat_mode": "LEGACY",
        "context_schema_version": "3",
        "context_budget_tokens": "13000",
    }

    context = bot._extract_context_metadata(payload)

    assert context["context_schema_version"] == 3
    assert context["context_contract_compat_mode"] == "legacy"
