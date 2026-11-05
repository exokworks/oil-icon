import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw


spec = importlib.util.spec_from_file_location('slice_icons', Path(__file__).parents[1] / 'scripts/slice_icons.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RemBgSessionTests(unittest.TestCase):
    def test_setup_and_slicing_default_to_the_skill_model_cache_and_cpu(self):
        new_session = Mock(return_value=object())
        with patch.dict(os.environ), patch.dict(sys.modules, {'rembg': SimpleNamespace(new_session=new_session)}):
            os.environ.pop('U2NET_HOME', None)
            os.environ.pop('REMBG_HOME', None)
            module.create_rembg_session()
            expected = Path(module.__file__).resolve().parents[1] / '.venv/models'
            self.assertEqual(Path(os.environ['U2NET_HOME']), expected)
        new_session.assert_called_once_with('u2net', providers=['CPUExecutionProvider'])

    def test_explicit_model_cache_is_preserved(self):
        for variable in ['U2NET_HOME', 'REMBG_HOME']:
            with self.subTest(variable=variable), patch.dict(os.environ), patch.dict(
                sys.modules, {'rembg': SimpleNamespace(new_session=Mock())}
            ):
                os.environ.pop('U2NET_HOME', None)
                os.environ.pop('REMBG_HOME', None)
                os.environ[variable] = '/custom/model/cache'
                module.create_rembg_session()
                self.assertEqual(os.environ[variable], '/custom/model/cache')
                if variable == 'REMBG_HOME':
                    self.assertNotIn('U2NET_HOME', os.environ)


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
                new_session=lambda _model, **_kwargs: object(),
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
