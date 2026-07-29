import base64
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_obs_420v.py")
SPEC = importlib.util.spec_from_file_location("audit_obs_420v", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class AuditObs420vTest(unittest.TestCase):
    def test_decodes_420v_internal_representation(self):
        text = "1920x1080 30.000-30.000 2 875704438"
        encoded = base64.b64encode(text.encode()).decode()

        description, value, fourcc = AUDIT.decode_supported_format(encoded)

        self.assertEqual(description, text)
        self.assertEqual(value, 875704438)
        self.assertEqual(fourcc, "420v")

    def test_finds_nested_capture_source_once(self):
        text = "1280x960 30.000-30.000 2 875704438"
        encoded = base64.b64encode(text.encode()).decode()
        document = {
            "sources": [
                {
                    "name": "Camera",
                    "settings": {"supported_format": encoded},
                    "duplicate": {
                        "name": "Camera",
                        "settings": {"supported_format": encoded},
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "Collection.json"
            scene.write_text(json.dumps(document), encoding="utf-8")

            formats = AUDIT.selected_formats(scene)

        self.assertEqual(len(formats), 1)
        self.assertEqual(formats[0].collection, "Collection")
        self.assertEqual(formats[0].source, "Camera")
        self.assertEqual(formats[0].fourcc, "420v")


if __name__ == "__main__":
    unittest.main()
