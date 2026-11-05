import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw


spec = importlib.util.spec_from_file_location('slice_icons', Path(__file__).parents[1] / 'scripts/slice_icons.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SliceIconsTests(unittest.TestCase):
    def test_rembg_soft_edge_preserves_original_rgba(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet = Path(directory) / 'sheet.png'
            outdir = Path(directory) / 'out'
            image = Image.new('RGB', (32, 32), (128, 128, 128))
            ImageDraw.Draw(image).rectangle((8, 8, 23, 23), fill=(200, 20, 30))
            image.save(sheet)

            alpha = Image.new('L', image.size)
            ImageDraw.Draw(alpha).rectangle((8, 8, 23, 23), fill=255)
            alpha.putpixel((16, 16), 128)
            rembg = SimpleNamespace(
                new_session=lambda _model: object(),
                remove=lambda cell, session: Image.merge('RGBA', (*cell.split(), alpha)),
            )
            with patch.dict(sys.modules, {'rembg': rembg}), patch.object(
                sys, 'argv', ['slice_icons', str(sheet), str(outdir), '--mode', 'rembg', '--grid', '1', '--size', '18']
            ):
                module.main()

            with Image.open(outdir / '01.png') as result:
                self.assertEqual(result.size, (18, 18))
                self.assertEqual(result.getpixel((9, 9)), (200, 20, 30, 128))
                self.assertEqual(result.getpixel((0, 0))[3], 0)
