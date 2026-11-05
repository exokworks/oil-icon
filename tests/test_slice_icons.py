import importlib.util
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
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


class BackgroundAndMatteTests(unittest.TestCase):
    def setUp(self):
        self.bg = (136, 136, 135)
        self.rgb = np.full((128, 128, 3), self.bg, dtype=np.uint8)
        self.rgb[24:104, 24:104] = (240, 210, 160)

    def test_measured_background_ignores_contaminated_corners(self):
        for y in (0, 118):
            for x in (0, 118):
                self.rgb[y:y+10, x:x+10] = (240, 210, 160)
        result = module.estimate_background(self.rgb)
        self.assertEqual(result['rgb'], list(self.bg))
        self.assertTrue(result['confident'])

    def test_background_noise_is_measured_without_erasing_similar_subject(self):
        noise = np.random.default_rng(7).integers(-1, 2, self.rgb.shape)
        rgb = (self.rgb.astype(int) + noise).astype(np.uint8)
        rgb[50:70, 50:70] = (148, 148, 140)
        background = module.estimate_background(rgb)
        alpha = module.floodfill_alpha(rgb, 24, background['rgb'], background['tolerance'])
        self.assertFalse(alpha[:20].any())
        self.assertTrue(np.all(alpha[50:70, 50:70] == 255))

    def test_background_drift_larger_than_old_threshold_is_removed(self):
        rgb = np.full((128, 128, 3), (170, 169, 168), dtype=np.uint8)
        rgb[30:90, 30:90] = (240, 210, 160)
        alpha = module.floodfill_alpha(rgb, 24)
        self.assertEqual(int(alpha[0, 0]), 0)
        self.assertEqual(int(alpha[50, 50]), 255)

    def test_enclosed_exact_background_is_removed_but_grey_subject_is_preserved(self):
        self.rgb[40:50, 40:50] = self.bg
        self.rgb[70:80, 70:80] = (148, 148, 140)
        alpha = module.floodfill_alpha(self.rgb, 30)
        self.assertTrue(np.all(alpha[40:50, 40:50] == 0))
        self.assertTrue(np.all(alpha[70:80, 70:80] == 255))

    def test_blended_edge_gets_partial_alpha_and_foreground_colour(self):
        self.rgb[24:104, 24] = (157, 151, 140)
        alpha = module.floodfill_alpha(self.rgb, 24)
        rgb, alpha = module.refine_edges(self.rgb, alpha, self.bg)
        self.assertAlmostEqual(int(alpha[60, 24]), 51, delta=2)
        np.testing.assert_allclose(rgb[60, 24], (240, 210, 160), atol=3)
        self.assertEqual(int(alpha[60, 25]), 255)
        np.testing.assert_array_equal(rgb[60, 25], self.rgb[60, 25])

    def test_rembg_matting_keeps_alpha_and_internal_transparency(self):
        self.rgb[24:104, 24] = (157, 151, 140)
        alpha = module.floodfill_alpha(self.rgb, 24)
        alpha[24:104, 24] = 51
        alpha[60, 60] = 128
        rgb, result = module.refine_edges(self.rgb, alpha, self.bg, infer_alpha=False)
        np.testing.assert_array_equal(result, alpha)
        np.testing.assert_array_equal(rgb[60, 60], self.rgb[60, 60])
        np.testing.assert_allclose(rgb[60, 24], (240, 210, 160), atol=3)


class ComponentOwnershipTests(unittest.TestCase):
    def test_detached_small_action_lines_are_preserved(self):
        alpha = np.zeros((256, 256), dtype=np.uint8)
        alpha[88:208, 68:172] = 255
        for y in (98, 118, 138):
            alpha[y:y+4, 192:222] = 255
        np.testing.assert_array_equal(module.clean(alpha), alpha)
        self.assertEqual(len(module.regions(module.clean(alpha) > 0)), 4)

    def test_low_opacity_and_diagonal_strokes_are_preserved(self):
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha[20:40, 20:40] = 255
        alpha[19, 20:40] = 16
        alpha[45, 45] = alpha[46, 46] = alpha[47, 47] = 255
        np.testing.assert_array_equal(module.clean(alpha), alpha)
        self.assertEqual(len(module.regions(alpha > 0)), 2)

    def test_neighbour_bleed_is_removed_using_original_cell_bounds(self):
        alpha = np.zeros((120, 120), dtype=np.uint8)
        alpha[30:80, 30:80] = 255
        alpha[2:7, 40:80] = 255  # wholly outside the original cell
        alpha[90:96, 0:12] = 255  # mostly outside and cut by the crop edge
        alpha[45:48, 95:103] = 255  # legitimate detached accent
        cleaned = module.clean(alpha, (10, 10, 110, 110))
        self.assertFalse(cleaned[:10].any())
        self.assertFalse(cleaned[90:96, :12].any())
        np.testing.assert_array_equal(cleaned[45:48, 95:103], alpha[45:48, 95:103])
        self.assertEqual(int(cleaned[50, 50]), 255)

    def test_faint_bridge_does_not_attach_neighbour_bleed_to_subject(self):
        alpha = np.zeros((120, 120), dtype=np.uint8)
        alpha[30:80, 30:80] = 255
        alpha[0:12, 40:60] = 255
        alpha[12:30, 45] = 1
        alpha[80, 30:80] = 16
        cleaned = module.clean(alpha, (10, 10, 110, 110))
        self.assertFalse(cleaned[:12].any())
        np.testing.assert_array_equal(cleaned[30:81, 30:80], alpha[30:81, 30:80])

    def test_detached_faint_accent_without_opaque_core_is_preserved(self):
        alpha = np.zeros((120, 120), dtype=np.uint8)
        alpha[30:80, 30:80] = 255
        alpha[90:94, 90:110] = 16
        np.testing.assert_array_equal(module.clean(alpha), alpha)
        faint_only = np.where(alpha > 0, 16, 0).astype(np.uint8)
        np.testing.assert_array_equal(module.clean(faint_only), faint_only)

    def test_ambiguous_overlap_is_kept_for_review(self):
        alpha = np.zeros((120, 120), dtype=np.uint8)
        alpha[5:12, 40:60] = 255
        np.testing.assert_array_equal(module.clean(alpha, (10, 10, 110, 110)), alpha)


class AutomaticQATests(unittest.TestCase):
    def test_reports_holes_grey_residue_and_cleanup_losses(self):
        rgb = np.full((80, 80, 3), (136, 136, 135), dtype=np.uint8)
        rgb[20:60, 20:60] = (240, 210, 160)
        rgb[22:26, 30:40] = (145, 145, 140)
        rgb[5:8, 30:40] = (240, 210, 160)
        before = module.floodfill_alpha(rgb)
        before[30:35, 30:35] = 0
        after = module.clean(before, (10, 10, 70, 70))
        report = module.inspect_cell(rgb, rgb, before, after, (10, 10, 70, 70), module.estimate_background(rgb), 24)
        codes = {finding['code'] for finding in report['findings']}
        self.assertTrue({'transparent_holes', 'background_like_foreground', 'removed_components'} <= codes)
        self.assertEqual(report['removed_pixels'], 30)

    def test_reports_components_lost_by_rembg_before_cleanup(self):
        rgb = np.full((128, 128, 3), (136, 136, 135), dtype=np.uint8)
        rgb[40:100, 30:80] = (240, 210, 160)
        rgb[50:54, 95:115] = (240, 210, 160)
        alpha = np.zeros((128, 128), dtype=np.uint8)
        alpha[40:100, 30:80] = 255
        report = module.inspect_cell(rgb, rgb, alpha, alpha, (0, 0, 128, 128), module.estimate_background(rgb), 24)
        losses = [finding for finding in report['findings'] if finding['code'] == 'possible_cutout_loss']
        self.assertEqual(len(losses), 1)
        self.assertEqual(losses[0]['pixels'], 80)

    def test_reports_attached_detail_loss_when_main_component_is_retained(self):
        rgb = np.full((128, 128, 3), (136, 136, 135), dtype=np.uint8)
        rgb[35:95, 30:80] = (240, 210, 160)
        rgb[54:57, 80:100] = (30, 80, 220)
        background = module.estimate_background(rgb)
        source = (module.colour_distance(rgb, background['rgb']) > background['tolerance'] + 2)
        self.assertEqual(len(module.regions(source)), 1)

        alpha = np.zeros((128, 128), dtype=np.uint8)
        alpha[35:95, 30:80] = 255
        report = module.inspect_cell(rgb, rgb, alpha, alpha, (0, 0, 128, 128), background, 24)
        losses = [finding for finding in report['findings'] if finding['code'] == 'possible_cutout_loss']
        self.assertEqual(len(losses), 1)
        self.assertEqual(losses[0]['pixels'], 60)
        self.assertEqual(losses[0]['regions'][0]['bbox'], [80, 54, 100, 57])

    def test_does_not_report_visible_semtransparent_detail_as_lost(self):
        rgb = np.full((128, 128, 3), (136, 136, 135), dtype=np.uint8)
        rgb[35:95, 30:80] = (240, 210, 160)
        rgb[54:57, 80:100] = (30, 80, 220)
        background = module.estimate_background(rgb)
        alpha = np.zeros((128, 128), dtype=np.uint8)
        alpha[35:95, 30:80] = 255
        alpha[54:57, 80:100] = 128
        report = module.inspect_cell(rgb, rgb, alpha, alpha, (0, 0, 128, 128), background, 24)
        self.assertNotIn('possible_cutout_loss', {finding['code'] for finding in report['findings']})

    def test_cli_writes_qa_for_all_16_icons_and_keeps_action_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet = Path(directory) / 'sheet.png'
            out = Path(directory) / 'out'
            source = Image.new('RGB', (1024, 1024), (136, 136, 135))
            draw = ImageDraw.Draw(source)
            for index in range(16):
                x, y = (index % 4) * 256, (index // 4) * 256
                draw.rectangle((x+68, y+88, x+171, y+207), fill=(240, 210, 160))
                for top in (98, 118, 138):
                    draw.rectangle((x+192, y+top, x+221, y+top+3), fill=(240, 210, 160))
                draw.rectangle((x+100, y+120, x+110, y+130), fill=(148, 148, 140))
            source.save(sheet)
            with patch.object(sys, 'argv', ['slice_icons', str(sheet), str(out), '--size', '256']), contextlib.redirect_stdout(io.StringIO()) as stdout:
                module.main()
            report = json.loads((out / 'qa/report.json').read_text())
            self.assertEqual(report['background']['hex'], '#888887')
            self.assertEqual(report['exported'], 16)
            self.assertEqual(report['review_cells'], list(range(1, 17)))
            self.assertEqual(len(list(out.glob('*.png'))), 16)
            for cell in report['cells']:
                self.assertEqual(cell['retained_components'], 4)
                codes = {finding['code'] for finding in cell['findings']}
                self.assertIn('background_like_foreground', codes)
                self.assertNotIn('transparent_holes', codes)
                self.assertNotIn('removed_components', codes)
            for name in ('source', 'cutout', 'cleaned', 'magenta', 'light', 'dark'):
                with Image.open(out / f'qa/{name}.png') as preview:
                    self.assertEqual(preview.size, (768, 864))
            self.assertNotIn('✓', stdout.getvalue())

    def test_partial_sheet_still_writes_reports_and_previews(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet = Path(directory) / 'sheet.png'
            out = Path(directory) / 'out'
            image = Image.new('RGB', (128, 128), '#888887')
            ImageDraw.Draw(image).rectangle((8, 8, 22, 22), fill='#f0d2a0')
            image.save(sheet)
            with patch.object(sys, 'argv', ['slice_icons', str(sheet), str(out), '--count', '1']), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            report = json.loads((out / 'qa/report.json').read_text())
            self.assertEqual(report['exported'], 1)
            self.assertEqual(report['skipped_blank'], 15)
            self.assertEqual(len(list(out.glob('*.png'))), 1)
            self.assertTrue((out / 'qa/magenta.png').is_file())

    def test_uneven_sheet_dimensions_keep_last_row_and_column(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet = Path(directory) / 'sheet.png'
            out = Path(directory) / 'out'
            image = Image.new('RGB', (100, 100), '#888887')
            ImageDraw.Draw(image).rectangle((90, 90, 99, 99), fill='#f0d2a0')
            image.save(sheet)
            with patch.object(sys, 'argv', ['slice_icons', str(sheet), str(out), '--grid', '3']), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            report = json.loads((out / 'qa/report.json').read_text())
            self.assertEqual(report['cells'][-1]['source_bounds'][2:], [100, 100])
            with Image.open(out / '09.png') as icon:
                self.assertIsNotNone(icon.getbbox())

    def test_uncertain_background_automatically_uses_rembg_and_warns(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet = Path(directory) / 'sheet.png'
            out = Path(directory) / 'out'
            rgb = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(sheet)
            remove = Mock(side_effect=lambda cell, session: cell.convert('RGBA'))
            rembg = SimpleNamespace(new_session=Mock(return_value=object()), remove=remove)
            with patch.dict(sys.modules, {'rembg': rembg}), patch.object(
                sys, 'argv', ['slice_icons', str(sheet), str(out), '--grid', '1']
            ), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            report = json.loads((out / 'qa/report.json').read_text())
            self.assertEqual(report['requested_mode'], 'floodfill')
            self.assertEqual(report['effective_mode'], 'rembg')
            self.assertFalse(report['background']['confident'])
            self.assertTrue(report['warnings'])
            remove.assert_called_once()
