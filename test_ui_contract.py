import os
import re
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "templates", "index.html")


class UiElementContractTests(unittest.TestCase):
    def test_literal_javascript_id_selectors_exist(self):
        with open(TEMPLATE, encoding="utf-8") as fh:
            html = fh.read()
        declared = set(re.findall(r'\bid=["\']([^"\']+)["\']', html))
        referenced = set(re.findall(r"\$\('#([^']+)'\)", html))
        missing = sorted(referenced - declared)
        self.assertEqual(missing, [], f"JavaScript references missing element IDs: {missing}")

    def test_settings_autofill_guards(self):
        with open(TEMPLATE, encoding="utf-8") as fh:
            html = fh.read()
        for field_id in ("rag-ai-model-identifier-v1", "rag-ai-credential-token-v1"):
            tag = re.search(rf'<input\b[^>]*\bid="{re.escape(field_id)}"[^>]*>', html)
            self.assertIsNotNone(tag, f"missing {field_id}")
            self.assertIn('data-bwignore="true"', tag.group(0))
            self.assertIn('data-1p-ignore="true"', tag.group(0))
            self.assertIn('data-lpignore="true"', tag.group(0))


if __name__ == "__main__":
    unittest.main()
