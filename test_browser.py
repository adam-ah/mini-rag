import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["SEARCH_OUTPUT"] = os.path.join(HERE, "tests", "small_corpus")
os.environ["SEARCH_INPUT"] = os.path.join(HERE, "tests", "small_corpus")

import app
from corpus import Corpus
from settings import SettingsService


try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class SettingsBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright Python package is not installed")
        cls.chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
        if not cls.chrome:
            raise unittest.SkipTest("Chrome/Chromium is not installed")

    def test_open_save_reopen_and_confirm_reset_without_console_errors(self):
        from werkzeug.serving import make_server

        with tempfile.TemporaryDirectory() as tmp:
            service = SettingsService(os.path.join(tmp, "settings.json"))
            ai = {
                "backend": "openai", "base_url": "http://127.0.0.1:1/v1",
                "model": "initial-browser-model", "api_key": "", "temperature": 0.2,
                "max_tokens": 2500, "timeout_seconds": 5,
            }
            service.save(ai, {"exclude_patterns": []})
            old_service = app.settings_service
            app.settings_service = service
            server = make_server("127.0.0.1", 0, app.app, threaded=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            errors = []
            try:
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(executable_path=self.chrome, headless=True)
                    page = browser.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")

                    page.locator("#settingsBtn").click()
                    modal = page.locator("#settingsModal")
                    modal.wait_for(state="visible")
                    model = page.locator("#rag-ai-model-identifier-v1")
                    self.assertEqual(model.input_value(), "initial-browser-model")
                    self.assertEqual(model.get_attribute("data-bwignore"), "true")
                    self.assertEqual(
                        page.locator("#rag-ai-credential-token-v1").get_attribute("autocomplete"),
                        "new-password",
                    )

                    model.fill("saved-browser-model")
                    page.locator("#set-save").click()
                    modal.wait_for(state="hidden")

                    page.locator("#settingsBtn").click()
                    self.assertEqual(model.input_value(), "saved-browser-model")
                    reset = page.locator("#settings-reset-to-defaults-v1")
                    reset.click()
                    self.assertEqual(reset.inner_text(), "Confirm reset")
                    self.assertTrue(modal.is_visible())
                    reset.click()
                    modal.wait_for(state="hidden")
                    browser.close()
            finally:
                server.shutdown()
                thread.join(timeout=5)
                app.settings_service = old_service

            self.assertEqual(errors, [], f"browser page errors: {errors}")

    def test_search_and_streamed_answer_workflows_against_sample_corpus(self):
        from werkzeug.serving import make_server

        sample = os.path.join(HERE, "tests", "big_corpus")
        sample_corpus = Corpus().load(text_dir=sample, source=sample)
        self.assertGreater(sample_corpus.N, 0)

        with tempfile.TemporaryDirectory() as tmp:
            service = SettingsService(os.path.join(tmp, "settings.json"))
            ai = {
                "backend": "openai", "base_url": "http://127.0.0.1:1/v1",
                "model": "browser-e2e-model", "api_key": "", "temperature": 0.2,
                "max_tokens": 2500, "timeout_seconds": 5,
            }
            service.save(ai, {"exclude_patterns": []})
            old_service = app.settings_service
            old_corpus = app.CORPUS[0]
            app.settings_service = service
            app.CORPUS[0] = sample_corpus
            def fake_stream(*_args, **_kwargs):
                yield "Ownership controls how values are moved."
                yield "\n\n"
                yield "Borrowing allows references without taking ownership [1]."

            stream_patch = mock.patch.object(app.backend, "stream", side_effect=fake_stream)
            connection_patch = mock.patch.object(
                app.backend, "test_connection", return_value=(True, "Connection successful")
            )
            stream_patch.start()
            connection_patch.start()
            server = make_server("127.0.0.1", 0, app.app, threaded=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            page_errors = []
            console_errors = []
            requests = []
            try:
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(executable_path=self.chrome, headless=True)
                    page = browser.new_page()
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.on(
                        "console",
                        lambda message: console_errors.append(message.text) if message.type == "error" else None,
                    )
                    page.on("request", lambda req: requests.append((req.method, req.url)))
                    page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")

                    question = page.locator("#q")
                    question.fill("ownership rules moving a value")
                    page.locator("#searchBtn").click()
                    first_hit = page.locator("#hits .hit").first
                    first_hit.wait_for(state="visible")
                    self.assertIn("ch04-01-what-is-ownership", first_hit.inner_text().lower())
                    self.assertGreaterEqual(page.locator("#hits .hit").count(), 1)

                    question.fill("term-that-does-not-exist-in-the-sample-corpus")
                    page.locator("#searchBtn").click()
                    no_matches = page.locator("#hits .empty")
                    no_matches.wait_for(state="visible")
                    self.assertEqual(no_matches.inner_text(), "No matches.")

                    question.fill("explain how ownership and borrowing work")
                    page.locator("#ask").click()
                    page.locator("#mode", has_text="mode: openai:browser-e2e-model").wait_for(state="visible")

                    self.assertGreaterEqual(page.locator("#steps > li").count(), 5)
                    self.assertIn("steps", page.locator("#stepsToggle").inner_text().lower())
                    answer = page.locator("#answer").inner_text()
                    self.assertIn("Ownership controls how values are moved.", answer)
                    self.assertIn("Borrowing allows references", answer)
                    self.assertGreater(page.locator("#sources > li").count(), 0)
                    self.assertIn("ownership", page.locator("#sources").inner_text().lower())
                    self.assertFalse(page.locator("#ask").is_disabled())

                    ask_requests = [(method, url) for method, url in requests if "/api/ask_stream" in url]
                    self.assertTrue(ask_requests)
                    self.assertTrue(all(method == "POST" for method, _ in ask_requests))
                    self.assertTrue(all("explain how" not in url for _, url in ask_requests))
                    browser.close()
            finally:
                server.shutdown()
                thread.join(timeout=5)
                stream_patch.stop()
                connection_patch.stop()
                app.settings_service = old_service
                app.CORPUS[0] = old_corpus

            self.assertEqual(page_errors, [], f"browser page errors: {page_errors}")
            self.assertEqual(console_errors, [], f"browser console errors: {console_errors}")

    def test_query_rescue_search_suggestions_and_streamed_answer(self):
        from werkzeug.serving import make_server

        fixture = os.path.join(HERE, "tests", "query_rescue_corpus")
        fixture_corpus = Corpus().load(text_dir=fixture, source=fixture)
        with tempfile.TemporaryDirectory() as tmp:
            service = SettingsService(os.path.join(tmp, "settings.json"))
            service.save({
                "backend": "openai", "base_url": "http://127.0.0.1:1/v1",
                "model": "browser-rescue-model", "api_key": "", "temperature": 0.2,
                "max_tokens": 2500, "timeout_seconds": 5,
            }, {"exclude_patterns": []})
            old_service = app.settings_service
            old_corpus = app.CORPUS[0]
            app.settings_service = service
            app.CORPUS[0] = fixture_corpus

            def fake_stream(*_args, **_kwargs):
                yield "Use 36 PSI at the front and 42 PSI at the rear [1]."

            stream_patch = mock.patch.object(app.backend, "stream", side_effect=fake_stream)
            connection_patch = mock.patch.object(
                app.backend, "test_connection", return_value=(True, "Connection successful")
            )
            stream_patch.start(); connection_patch.start()
            server = make_server("127.0.0.1", 0, app.app, threaded=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            page_errors, console_errors = [], []
            try:
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(executable_path=self.chrome, headless=True)
                    page = browser.new_page()
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.on("console", lambda message:
                            console_errors.append(message.text) if message.type == "error" else None)
                    page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")

                    question = page.locator("#q")
                    question.fill("what pressure do my tyres need?")
                    page.locator("#searchBtn").click()
                    first_hit = page.locator("#hits .hit").first
                    first_hit.wait_for(state="visible")
                    self.assertIn("tyre_specs", first_hit.inner_text())
                    self.assertIn("36 psi", first_hit.inner_text().lower())
                    self.assertIn("42 psi", first_hit.inner_text().lower())
                    self.assertIn("tyre → tire", page.locator("#queryExpansionNotice").inner_text())
                    suggestion = page.locator("#querySuggestions button").first
                    self.assertTrue(suggestion.is_visible())
                    with page.expect_request(lambda req: "/api/search" in req.url):
                        suggestion.click()
                    self.assertIn("tire", question.input_value().lower())

                    question.fill("what pressure do my tyres need?")
                    page.locator("#ask").click()
                    page.locator("#mode", has_text="browser-rescue-model").wait_for(state="visible")
                    page.locator("#stepsToggle").click()
                    self.assertIn("Expanded search: tyre → tire", page.locator("#stepsCard").inner_text())
                    self.assertIn("36 PSI", page.locator("#answer").inner_text())
                    self.assertIn("tyre_specs", page.locator("#sources").inner_text())
                    self.assertTrue(page.locator("#querySuggestions button").first.is_visible())

                    question.fill("a query with no fixture matches whatsoever zzzqqq")
                    page.locator("#searchBtn").click()
                    page.locator("#hits .empty").wait_for(state="visible")
                    self.assertEqual(page.locator("#sources > li").count(), 0)
                    self.assertFalse(page.locator("#sourcesCard").is_visible())
                    browser.close()
            finally:
                server.shutdown(); thread.join(timeout=5)
                stream_patch.stop(); connection_patch.stop()
                app.settings_service = old_service
                app.CORPUS[0] = old_corpus

            self.assertEqual(page_errors, [], f"browser page errors: {page_errors}")
            self.assertEqual(console_errors, [], f"browser console errors: {console_errors}")


if __name__ == "__main__":
    unittest.main()
