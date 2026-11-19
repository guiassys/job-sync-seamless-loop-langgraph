import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from langchain_core.messages.block_translators.openai import (
    convert_to_openai_image_block,
)
from src.reference_image import encode_reference_image


class ReferenceImageTests(unittest.TestCase):
    def test_reference_image_encodes_as_supported_rgb_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            Image.new("RGB", (1600, 900), color=(30, 120, 180)).save(path)

            encoded = encode_reference_image(path)

        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (1024, 576))

    def test_missing_reference_image_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.png"
            with self.assertRaises(FileNotFoundError):
                encode_reference_image(path)

    @patch("src.nodes.vision_llm")
    def test_director_sends_reference_image_to_vision_model(self, vision_llm):
        from src.nodes import REFERENCE_IMAGE_PATH, director_node

        vision_llm.invoke.return_value = SimpleNamespace(
            content="Lo-fi anime illustration, rainy night and teal-orange lighting."
        )

        result = director_node({"user_prompt": "Crie uma cena urbana com chuva."})

        messages = vision_llm.invoke.call_args.args[0]
        content = messages[1].content
        image_blocks = [part for part in content if part.get("type") == "image"]

        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]["mime_type"], "image/jpeg")
        self.assertTrue(REFERENCE_IMAGE_PATH.is_file())
        self.assertIn(str(REFERENCE_IMAGE_PATH), content[0]["text"])
        self.assertIn("Lo-fi anime illustration", result["visual_specification"])
        self.assertEqual(
            convert_to_openai_image_block(image_blocks[0])["type"], "image_url"
        )
        with Image.open(
            io.BytesIO(base64.b64decode(image_blocks[0]["base64"]))
        ) as image:
            self.assertEqual(image.format, "JPEG")


if __name__ == "__main__":
    unittest.main()
