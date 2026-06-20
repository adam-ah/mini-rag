import json
import os
import stat
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["SEARCH_OUTPUT"] = os.path.join(HERE, "tests", "small_corpus")
os.environ["SEARCH_INPUT"] = os.path.join(HERE, "tests", "small_corpus")

from settings import SettingsService
from env_loader import DOTENV_VALUES


class SettingsPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "nested", "settings.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_creates_file_and_normalizes_url(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            service = SettingsService(self.path)
            ai = {
                "backend": "openai", "base_url": "http://localhost:8080/v1/",
                "model": "local", "api_key": "secret", "temperature": 0.2,
                "max_tokens": 2500, "timeout_seconds": 180,
            }
            service.save(ai, {"exclude_patterns": []})

        with open(self.path, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(saved["ai"]["base_url"], "http://localhost:8080/v1")
        self.assertEqual(service.get().ai.base_url, "http://localhost:8080/v1")
        self.assertNotIn("secret", json.dumps(service.get_public()))
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_failed_replace_keeps_previous_runtime_settings(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            service = SettingsService(self.path)
            before = service.get()
            ai = {
                "backend": "openai", "base_url": "http://different.invalid/v1",
                "model": "changed", "api_key": "", "temperature": 0.2,
                "max_tokens": 2500, "timeout_seconds": 180,
            }
            with mock.patch("settings.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    service.save(ai, {"exclude_patterns": []})
            self.assertEqual(service.get(), before)

    def test_gui_save_overrides_dotenv_value(self):
        dotenv_url = "http://dotenv.example/v1"
        gui_url = "http://gui.example/v1"
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": dotenv_url}, clear=True), \
             mock.patch.dict(DOTENV_VALUES, {"OPENAI_BASE_URL": dotenv_url}, clear=True):
            service = SettingsService(self.path)
            self.assertEqual(service.get().ai.base_url, dotenv_url)
            ai = {
                "backend": "openai", "base_url": gui_url, "model": "local",
                "api_key": "", "temperature": 0.2, "max_tokens": 2500,
                "timeout_seconds": 180,
            }
            service.save(ai, {"exclude_patterns": []})
            self.assertEqual(service.get().ai.base_url, gui_url)
            self.assertNotIn("ai.base_url", service.get_public()["overridden"])

    def test_real_process_environment_still_overrides_gui_save(self):
        env_url = "http://managed.example/v1"
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": env_url}, clear=True), \
             mock.patch.dict(DOTENV_VALUES, {}, clear=True):
            service = SettingsService(self.path)
            ai = {
                "backend": "openai", "base_url": "http://gui.example/v1", "model": "local",
                "api_key": "", "temperature": 0.2, "max_tokens": 2500,
                "timeout_seconds": 180,
            }
            service.save(ai, {"exclude_patterns": []})
            self.assertEqual(service.get().ai.base_url, env_url)
            self.assertIn("ai.base_url", service.get_public()["overridden"])

    def test_reset_removes_saved_file_and_restores_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            service = SettingsService(self.path)
            ai = {
                "backend": "extractive", "base_url": "http://changed.example/v1",
                "model": "changed", "api_key": "secret", "temperature": 0.8,
                "max_tokens": 999, "timeout_seconds": 99,
            }
            service.save(ai, {"exclude_patterns": ["private/**"]})
            self.assertTrue(os.path.exists(self.path))
            service.reset()
            self.assertFalse(os.path.exists(self.path))
            self.assertEqual(service.get(), SettingsService(os.path.join(self.tmp.name, "other.json")).get())
            self.assertFalse(service.get_public()["api_key_configured"])

    def test_legacy_alias_setting_is_ignored_without_breaking_load(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({
                "version": 1,
                "ai": {},
                "retrieval": {"exclude_patterns": ["private/**"], "aliases": {"tyre": "tire"}},
            }, fh)
        with mock.patch.dict(os.environ, {}, clear=True):
            service = SettingsService(self.path)
        self.assertEqual(service.get().retrieval.exclude_patterns, ["private/**"])
        self.assertNotIn("aliases", service.get_public()["settings"]["retrieval"])


class SettingsApiTests(unittest.TestCase):
    def test_gui_shaped_payload_saves(self):
        import app

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {}, clear=True):
                service = SettingsService(os.path.join(tmp, "settings.json"))
            old = app.settings_service
            app.settings_service = service
            try:
                response = app.app.test_client().put(
                    "/api/settings",
                    json={"ai": {
                        "backend": "openai", "base_url": "http://localhost:8080/v1",
                        "model": "local", "temperature": 0.2,
                        "max_tokens": 2500, "timeout_seconds": 180,
                    }, "retrieval": {}},
                    headers={"X-CSRF-Token": app.CSRF_TOKEN},
                )
            finally:
                app.settings_service = old
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            self.assertEqual(response.get_json(), {"status": "saved"})

    def test_reset_requires_confirmation_then_resets(self):
        import app

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            service = SettingsService(os.path.join(tmp, "settings.json"))
            old = app.settings_service
            app.settings_service = service
            try:
                client = app.app.test_client()
                missing = client.delete(
                    "/api/settings", json={},
                    headers={"X-CSRF-Token": app.CSRF_TOKEN},
                )
                confirmed = client.delete(
                    "/api/settings", json={"confirm": "RESET"},
                    headers={"X-CSRF-Token": app.CSRF_TOKEN},
                )
            finally:
                app.settings_service = old
            self.assertEqual(missing.status_code, 400)
            self.assertEqual(confirmed.status_code, 200)
            self.assertEqual(confirmed.get_json(), {"status": "reset"})


if __name__ == "__main__":
    unittest.main()
