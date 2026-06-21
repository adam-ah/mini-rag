import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import convert


class ConversionSyncTests(unittest.TestCase):
    def test_pdf_without_extractable_text_is_not_reconverted(self):
        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "input")
            output_dir = os.path.join(root, "output")
            os.makedirs(input_dir)
            source = os.path.join(input_dir, "scan.pdf")
            with open(source, "wb") as handle:
                handle.write(b"synthetic scan")
            with mock.patch.object(convert, "INPUT", input_dir), \
                 mock.patch.object(convert, "OUTPUT", output_dir), \
                 mock.patch.object(convert, "conv_pdf", return_value="") as convert_pdf:
                first = convert.run_conversion()
                second = convert.run_conversion()
            self.assertEqual(first.converted, 1)
            self.assertEqual(second.converted, 0)
            self.assertEqual(second.skipped, 1)
            self.assertEqual(convert_pdf.call_count, 1)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "scan.txt")))
            self.assertEqual([error.error_category for error in second.errors], ["OCR_NEEDED"])
            output = StringIO()
            with mock.patch.object(convert, "run_conversion", return_value=second), \
                 redirect_stdout(output):
                convert.main()
            self.assertIn("WARN: scan.pdf: No extractable text", output.getvalue())


class TestScriptTests(unittest.TestCase):
    def test_live_ai_uses_dedicated_endpoint_by_default(self):
        path = os.path.join(os.path.dirname(__file__), "test.sh")
        with open(path, encoding="utf-8") as handle:
            script = handle.read()
        self.assertIn('export RUN_LIVE_AI_TESTS="${RUN_LIVE_AI_TESTS:-1}"', script)
        self.assertIn(
            'export AI_TEST_BASE_URL="${AI_TEST_BASE_URL:-http://10.0.0.10:8080/v1}"',
            script,
        )


if __name__ == "__main__":
    unittest.main()
