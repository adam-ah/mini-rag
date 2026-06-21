import os, json, threading, tempfile
from dataclasses import dataclass, asdict, field, replace
from typing import Dict, List, Optional, Any
from env_loader import DOTENV_VALUES, load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

@dataclass(frozen=True)
class AISettings:
    backend: str = "openai"
    base_url: str = "http://localhost:8080/v1"
    model: str = "local"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 2500
    timeout_seconds: int = 180
    adaptive_refinement: bool = True
    reflection_max_queries: int = 3
    reflection_max_tokens: int = 400
    reflection_context_budget: int = 32000
    refinement_context_budget: int = 10000

@dataclass(frozen=True)
class RetrievalSettings:
    exclude_patterns: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Settings:
    version: int = 1
    ai: AISettings = field(default_factory=AISettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)

class SettingsService:
    def __init__(self, filepath: Optional[str] = None):
        configured = filepath or os.environ.get("SEARCH_SETTINGS_FILE") or os.path.join(HERE, "settings.json")
        self.filepath = os.path.abspath(configured)
        self._lock = threading.Lock()
        self._current: Settings = Settings()
        self._overridden: Dict[str, Any] = {}
        self._file_ai_fields = set()
        self.load()

    def load(self):
        with self._lock:
            try:
                if os.path.exists(self.filepath):
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if data.get("version", 1) != 1:
                        raise ValueError(f"Unsupported settings version: {data.get('version')}")

                    # Simple merge: load file, then apply env
                    ai_data = data.get("ai", {})
                    # Ignore retired retrieval keys from older settings files.
                    ret_data = data.get("retrieval", {})
                    ret_data = {"exclude_patterns": ret_data.get("exclude_patterns", [])}
                    self._file_ai_fields = set(ai_data)

                    self._current = Settings(
                        version=data.get("version", 1),
                        ai=AISettings(**ai_data),
                        retrieval=RetrievalSettings(**ret_data)
                    )
                else:
                    self._current = Settings()
                    self._file_ai_fields = set()
            except Exception as e:
                # Treat malformed JSON as error, but we'll let the app handle it.
                # For now, just print and use defaults.
                print(f"Warning: Failed to load settings: {e}")
                self._current = Settings()

            self._apply_env_overrides()

    def _apply_env_overrides(self):
        # Mapping of env var to (category, field)
        mapping = {
            "SEARCH_BACKEND": ("ai", "backend"),
            "OPENAI_BASE_URL": ("ai", "base_url"),
            "SEARCH_MODEL": ("ai", "model"),
            "OPENAI_API_KEY": ("ai", "api_key"),
            "SEARCH_TEMPERATURE": ("ai", "temperature"),
            "SEARCH_MAX_TOKENS": ("ai", "max_tokens"),
            "SEARCH_TIMEOUT": ("ai", "timeout_seconds"),
            "SEARCH_ADAPTIVE_REFINEMENT": ("ai", "adaptive_refinement"),
            "SEARCH_REFLECTION_MAX_QUERIES": ("ai", "reflection_max_queries"),
            "SEARCH_REFLECTION_MAX_TOKENS": ("ai", "reflection_max_tokens"),
            "SEARCH_REFLECTION_CONTEXT_BUDGET": ("ai", "reflection_context_budget"),
            "SEARCH_REFINEMENT_CONTEXT_BUDGET": ("ai", "refinement_context_budget"),
        }

        overridden = {}
        current_ai = asdict(self._current.ai)
        current_ret = asdict(self._current.retrieval)

        for env, (cat, field_name) in mapping.items():
            val = os.environ.get(env)
            if val is not None:
                # settings.json contains GUI choices. They override values loaded
                # from .env; genuine process environment variables still win.
                from_dotenv = DOTENV_VALUES.get(env) == val
                if from_dotenv and field_name in self._file_ai_fields:
                    continue
                overridden[f"{cat}.{field_name}"] = val
                if cat == "ai":
                    # Cast types
                    if field_name == "temperature": val = float(val)
                    elif field_name == "adaptive_refinement":
                        val = str(val).strip().lower() in ("1", "true", "yes", "on")
                    elif field_name in ("max_tokens", "timeout_seconds", "reflection_max_queries",
                                         "reflection_max_tokens", "reflection_context_budget",
                                         "refinement_context_budget"):
                        val = int(val)
                    current_ai[field_name] = val

        self._current = Settings(
            version=self._current.version,
            ai=AISettings(**current_ai),
            retrieval=RetrievalSettings(**current_ret)
        )
        self._overridden = overridden

    def get(self) -> Settings:
        with self._lock:
            return self._current

    def get_public(self) -> Dict[str, Any]:
        with self._lock:
            s = asdict(self._current)
            s["ai"]["api_key"] = ""
            return {
                "settings": s,
                "api_key_configured": bool(self._current.ai.api_key),
                "overridden": self._overridden
            }

    def save(self, new_ai: Dict[str, Any], new_ret: Dict[str, Any]):
        with self._lock:
            # Validation
            ai = AISettings(**new_ai)
            ret = RetrievalSettings(**new_ret)

            # Validate values
            if ai.backend not in ("openai", "claude", "extractive"):
                raise ValueError("Invalid backend")
            if not ai.base_url.startswith(("http://", "https://")):
                raise ValueError("Invalid base_url")
            if not ai.model and ai.backend != "extractive":
                raise ValueError("Model is required for AI backends")
            if not (0 <= ai.temperature <= 2):
                raise ValueError("Temperature must be between 0 and 2")
            if not (1 <= ai.max_tokens <= 32768):
                raise ValueError("max_tokens must be between 1 and 32768")
            if not (1 <= ai.timeout_seconds <= 600):
                raise ValueError("timeout_seconds must be between 1 and 600")
            if not isinstance(ai.adaptive_refinement, bool):
                raise ValueError("adaptive_refinement must be a boolean")
            if not (1 <= ai.reflection_max_queries <= 5):
                raise ValueError("reflection_max_queries must be between 1 and 5")
            if not (64 <= ai.reflection_max_tokens <= 2000):
                raise ValueError("reflection_max_tokens must be between 64 and 2000")
            if not (1000 <= ai.reflection_context_budget <= 100000):
                raise ValueError("reflection_context_budget must be between 1000 and 100000")
            if not (1000 <= ai.refinement_context_budget <= 100000):
                raise ValueError("refinement_context_budget must be between 1000 and 100000")

            # Normalize base_url
            normalized_url = ai.base_url.rstrip("/")
            ai = replace(ai, base_url=normalized_url)

            # Atomic save
            data = {
                "version": 1,
                "ai": asdict(ai),
                "retrieval": asdict(ret)
            }

            settings_dir = os.path.dirname(self.filepath)
            os.makedirs(settings_dir, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=settings_dir)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    os.chmod(temp_path, 0o600)
                except OSError:
                    pass
                os.replace(temp_path, self.filepath)
            except Exception:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

            # Publish only after persistence succeeds, so a failed save leaves the
            # last known-good runtime settings active.
            self._current = Settings(ai=ai, retrieval=ret)
            self._file_ai_fields = set(new_ai)
            self._apply_env_overrides()

    def reset(self):
        """Remove GUI overrides and return to .env/application defaults."""
        with self._lock:
            try:
                os.remove(self.filepath)
            except FileNotFoundError:
                pass
            self._current = Settings()
            self._file_ai_fields = set()
            self._apply_env_overrides()

# Global instance
settings_service = SettingsService()
