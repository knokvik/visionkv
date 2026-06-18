"""Tests for integration harness helpers."""

from __future__ import annotations

import tempfile
import unittest

from visionkv.integration_harness import _to_image_url, build_multimodal_message


class IntegrationHarnessTests(unittest.TestCase):
    def test_local_image_is_encoded_as_data_url(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            handle.write(b"fakepngbytes")
            handle.flush()
            data_url = _to_image_url(handle.name)

        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_message_contains_text_and_image_parts(self) -> None:
        message = build_multimodal_message("Describe the image.", "https://example.com/cat.png")
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"][0]["text"], "Describe the image.")
        self.assertEqual(
            message["content"][1]["image_url"]["url"],
            "https://example.com/cat.png",
        )


if __name__ == "__main__":
    unittest.main()
