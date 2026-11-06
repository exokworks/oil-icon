import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


spec = importlib.util.spec_from_file_location('slice_icons', Path(__file__).parents[1] / 'scripts/slice_icons.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CutoutRepairTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bg = np.array([136, 136, 135])
        noise = np.random.default_rng(7).integers(-2, 3, (128, 128, 3))
        self.rgb = (self.bg + noise).astype(np.uint8)
        self.rgb[24:104, 24:104] = (240, 210, 160)

    def run_cli(self, rgb, *args, name='out', repairs=None):
        sheet = self.root / 'sheet.png'
        Image.fromarray(rgb).save(sheet)
        out = self.root / name
        argv = ['slice_icons', str(sheet), str(out), '--grid', '1', '--size', '128', *args]
        if repairs is not None:
            path = self.root / 'repairs.json'
            path.write_text(json.dumps(repairs))
            argv += ['--repairs', str(path)]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        return out, json.loads((out / 'qa/report.json').read_text())

    def test_exterior_tolerance_removes_wide_halo_but_preserves_enclosed_grey(self):
        self.rgb[18:110, 18:110] = self.bg + 18
        self.rgb[24:104, 24:104] = (240, 210, 160)
        self.rgb[50:70, 50:70] = self.bg + 18
        background = module.estimate_background(self.rgb)
        self.assertEqual(background['tolerance'], 3)
        halo = np.zeros((128, 128), bool)
        halo[18:110, 18:110] = True
        halo[24:104, 24:104] = False
        default = module.floodfill_alpha(self.rgb, 32, background['rgb'], background['tolerance'])
        repaired = module.floodfill_alpha(self.rgb, 24, background['rgb'], background['tolerance'], 32)
        self.assertTrue(np.all(default[halo] == 255))
        self.assertFalse(repaired[halo].any())
        self.assertTrue(np.all(repaired[50:70, 50:70] == 255))
        self.assertEqual(repaired[40, 40], 255)

    def test_cli_records_exterior_override_without_changing_sample_tolerance(self):
        self.rgb[18:110, 18:110] = self.bg + 18
        self.rgb[24:104, 24:104] = (240, 210, 160)
        original, first = self.run_cli(self.rgb, '--thresh', '32', name='original')
        corrected, second = self.run_cli(self.rgb, '--background-tolerance', '32')
        self.assertEqual(first['background']['tolerance'], 3)
        self.assertEqual(second['background']['tolerance'], 3)
        self.assertEqual(first['cells'][0]['background_tolerance'], 3)
        self.assertEqual(second['cells'][0]['background_tolerance'], 32)
        with Image.open(original / '01.png') as image:
            before = np.asarray(image)
        with Image.open(corrected / '01.png') as image:
            after = np.asarray(image)
        near_grey_before = (module.colour_distance(before[:, :, :3], self.bg) < 24) & (before[:, :, 3] > 128)
        near_grey_after = (module.colour_distance(after[:, :, :3], self.bg) < 24) & (after[:, :, 3] > 128)
        self.assertGreater(np.count_nonzero(near_grey_before), 100)
        self.assertFalse(near_grey_after.any())

    def test_zero_exterior_tolerance_keeps_exterior_accent_but_clears_enclosed_hole(self):
        rgb = np.full((128, 128, 3), self.bg, dtype=np.uint8)
        rgb[24:104, 24:104] = (240, 210, 160)
        rgb[58:63, 104:115] = self.bg + 1
        rgb[40:50, 40:50] = self.bg + 1
        alpha = module.floodfill_alpha(rgb, bg=self.bg, tolerance=3, background_tolerance=0)
        self.assertFalse(alpha[:20].any())
        self.assertTrue(np.all(alpha[58:63, 104:115] == 255))
        self.assertFalse(alpha[40:50, 40:50].any())

        # Exercise argument parsing, matting, cleanup and the exported PNG too.
        # A 91px bounding box produces a 105px canvas without resampling.
        out, report = self.run_cli(rgb, '--background-tolerance', '0', '--size', '105')
        self.assertEqual(report['cells'][0]['background_tolerance'], 0)
        with Image.open(out / '01.png') as image:
            result = np.asarray(image)
        self.assertTrue(np.all(result[46:51, 87:98, 3] == 255))
        self.assertFalse(result[28:38, 23:33, 3].any())

    def test_narrow_exterior_tolerance_does_not_classify_noisy_outside_as_holes(self):
        # Near-exact pixels remain connected to the edge through sampled noise.
        rgb = np.full((32, 32, 3), self.bg + 2, dtype=np.uint8)
        rgb[10:20, 10:20] = self.bg + 1
        for tolerance in (0, 1):
            with self.subTest(tolerance=tolerance):
                alpha = module.floodfill_alpha(rgb, bg=self.bg, tolerance=3, background_tolerance=tolerance)
                self.assertTrue(np.all(alpha == 255))

    def test_local_erasure_clears_only_seeded_component_within_bounds(self):
        self.rgb[40:60, 40:60] = self.bg + 2
        self.rgb[65:75, 40:50] = self.bg + 2  # another enclosed grey detail
        self.rgb[80:90, 80:90] = self.bg + 2
        alpha = module.floodfill_alpha(self.rgb, bg=self.bg, tolerance=3)
        self.assertEqual(np.count_nonzero(alpha[40:60, 40:60]), 400)
        erasures = [{'seed': [45, 45], 'bounds': [35, 35, 75, 78], 'tolerance': 3}]
        repaired, log = module.erase_background_regions(self.rgb, alpha, self.bg, erasures)
        self.assertFalse(repaired[40:60, 40:60].any())
        self.assertTrue(np.all(repaired[65:75, 40:50] == 255))
        self.assertTrue(np.all(repaired[80:90, 80:90] == 255))
        self.assertEqual(repaired[39, 45], 255)
        self.assertEqual(log[0]['removed_pixels'], 400)
        np.testing.assert_array_equal(alpha[40:60, 40:60], 255)

    def test_local_erasure_cannot_spread_outside_selected_rectangle(self):
        self.rgb[40:80, 40:80] = self.bg + 2
        alpha = module.floodfill_alpha(self.rgb, bg=self.bg, tolerance=3)
        repaired, _ = module.erase_background_regions(self.rgb, alpha, self.bg, [
            {'seed': [45, 45], 'bounds': [40, 40, 60, 60], 'tolerance': 3},
        ])
        self.assertFalse(repaired[40:60, 40:60].any())
        self.assertTrue(np.all(repaired[60:80, 40:80] == 255))
        self.assertTrue(np.all(repaired[40:60, 60:80] == 255))

    def test_cli_local_repairs_are_applied_and_logged(self):
        self.rgb[40:60, 40:60] = self.bg + 2
        self.rgb[70:80, 70:80] = self.bg + 2
        out, report = self.run_cli(self.rgb, repairs={'cells': {'1': {'erase_regions': [
            {'seed': [45, 45], 'bounds': [38, 38, 62, 62], 'tolerance': 3},
        ]}}})
        entry = report['cells'][0]
        self.assertEqual(entry['applied_repairs'][0]['removed_pixels'], 400)
        self.assertIn('transparent_holes', [item['code'] for item in entry['findings']])
        with Image.open(out / '01.png') as image:
            self.assertIsNotNone(image.getbbox())

    def test_cli_local_repair_preserves_surrounding_art_and_candidate(self):
        self.rgb[38:62, 38:62] = (157, 151, 140)
        self.rgb[40:60, 40:60] = self.bg + 2
        repairs = {'cells': {'1': {'erase_regions': [
            {'seed': [45, 45], 'bounds': [40, 40, 60, 60], 'tolerance': 3},
        ]}}}

        def remove(cell, session):
            rgb = np.asarray(cell)
            alpha = module.floodfill_alpha(rgb, bg=self.bg, tolerance=3)
            return Image.fromarray(np.dstack((rgb, alpha)))

        rembg = SimpleNamespace(new_session=Mock(return_value=object()), remove=remove)
        for mode in ('floodfill', 'rembg'):
            with self.subTest(mode=mode), patch.dict(sys.modules, {'rembg': rembg}):
                before_dir, _ = self.run_cli(self.rgb, '--mode', mode, '--size', '92', name=f'{mode}-before')
                after_dir, report = self.run_cli(self.rgb, '--mode', mode, '--size', '92', name=f'{mode}-after', repairs=repairs)
                with Image.open(before_dir / '01.png') as image:
                    before = np.asarray(image)
                with Image.open(after_dir / '01.png') as image:
                    after = np.asarray(image)
                # Same 80px subject box and 92px canvas: no resize interpolation.
                outside = np.ones((92, 92), dtype=bool)
                outside[22:42, 22:42] = False
                np.testing.assert_array_equal(after[outside], before[outside])
                np.testing.assert_array_equal(after[21, 27], [157, 151, 140, 255])
                self.assertFalse(after[22:42, 22:42, 3].any())
                self.assertEqual(report['cells'][0]['applied_repairs'][0]['removed_pixels'], 400)
                if mode == 'rembg':
                    with Image.open(before_dir / 'qa/colour-candidates/01.png') as image:
                        candidate_before = np.asarray(image)
                    with Image.open(after_dir / 'qa/colour-candidates/01.png') as image:
                        candidate_after = np.asarray(image)
                    outside = np.ones((128, 128), dtype=bool)
                    outside[40:60, 40:60] = False
                    np.testing.assert_array_equal(candidate_after[outside], candidate_before[outside])
                    self.assertFalse(candidate_after[40:60, 40:60, 3].any())

    def test_cli_local_rembg_repair_preserves_existing_soft_edge(self):
        self.rgb[38:62, 38:62] = (157, 151, 140)
        self.rgb[40:60, 40:60] = self.bg + 2

        def remove(cell, session):
            rgb = np.asarray(cell)
            alpha = module.floodfill_alpha(rgb, bg=self.bg, tolerance=3)
            alpha[38:62, 38:62] = 52
            alpha[40:60, 40:60] = 255
            return Image.fromarray(np.dstack((rgb, alpha)))

        rembg = SimpleNamespace(new_session=Mock(return_value=object()), remove=remove)
        with patch.dict(sys.modules, {'rembg': rembg}):
            out, _ = self.run_cli(self.rgb, '--mode', 'rembg', '--size', '92', repairs={'cells': {'1': {'erase_regions': [
                {'seed': [45, 45], 'bounds': [40, 40, 60, 60], 'tolerance': 3},
            ]}}})
        with Image.open(out / '01.png') as image:
            np.testing.assert_array_equal(np.asarray(image)[21, 27], [157, 151, 140, 52])

    def test_duplicate_repair_keys_are_rejected_before_export_or_overwrite(self):
        cases = [
            ('{"cells":{"1":{"erase_regions":[]},"1":{"background_tolerance":32}}}', '1'),
            ('{"cells":{},"cells":{"1":{}}}', 'cells'),
            ('{"cells":{"1":{"mode":"floodfill","mode":"rembg"}}}', 'mode'),
            ('{"cells":{"1":{"erase_regions":[{"seed":[1,1],"seed":[2,2],"bounds":[0,0,10,10],"tolerance":3}]}}}', 'seed'),
        ]
        for index, (text, key) in enumerate(cases):
            for existing in (False, True):
                with self.subTest(key=key, existing=existing):
                    name = f'duplicate-{index}-{existing}'
                    out = self.root / name
                    if existing:
                        self.run_cli(self.rgb, name=name)
                        snapshot = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*') if p.is_file()}
                    path = self.root / 'duplicate.json'
                    path.write_text(text)
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                        self.run_cli(self.rgb, '--repairs', str(path), name=name)
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn(f'duplicate repair key: {key!r}', stderr.getvalue())
                    if existing:
                        current = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*') if p.is_file()}
                        self.assertEqual(current, snapshot)
                    else:
                        self.assertFalse(out.exists())

    def test_invalid_repairs_fail_before_export(self):
        cases = [
            {'cells': {'0': {}}},
            {'cells': {'01': {}}},
            {'cells': {'2': {}}},
            {'cells': {'1': {'mode': 'magic'}}},
            {'cells': {'1': {'background_tolerance': 256}}},
            {'cells': {'1': {'background_tolerance': True}}},
            {'cells': {'1': {'erase_regions': [{'seed': [45, 45], 'bounds': [40, 40, 60, 60], 'tolerance': 3}]}}},
            {'cells': {'1': {'erase_regions': [{'seed': [1, 1], 'bounds': [-1, 0, 10, 10], 'tolerance': 3}]}}},
            {'cells': {'1': {'erase_regions': [{'seed': [1, 1], 'bounds': [0, 0, 10, 10], 'tolerance': -1}]}}},
            {'cells': {'1': {'erase_regions': [{'seed': [1, 1], 'bounds': [0, 0, 10, 10]}]}}},
            {'cells': [], 'unexpected': True},
        ]
        for index, repairs in enumerate(cases):
            with self.subTest(repairs=repairs), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.run_cli(self.rgb, repairs=repairs, name=f'bad-{index}')
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((self.root / f'bad-{index}').exists())

    def test_repairs_cannot_target_neighbour_padding(self):
        image = Image.fromarray(self.rgb)
        path = self.root / 'padding.json'
        path.write_text(json.dumps({'cells': {'2': {'erase_regions': [
            {'seed': [0, 0], 'bounds': [0, 0, 2, 2], 'tolerance': 255},
        ]}}}))
        with self.assertRaisesRegex(ValueError, 'owned_bounds'):
            module.load_repairs(path, image, 2, 4, self.bg)

    def test_invalid_exterior_tolerance_fails_before_export(self):
        for value in ('-1', '256'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_cli(self.rgb, '--background-tolerance', value)
            self.assertFalse((self.root / 'out').exists())

    def test_uncertain_rembg_loss_is_flagged_and_colour_candidate_can_be_selected(self):
        self.rgb[:4] = np.random.default_rng(1).integers(0, 256, (4, 128, 3), dtype=np.uint8)
        self.rgb[-4:] = 40
        self.rgb[:, :4] = 200
        self.rgb[:, -4:] = 80
        self.rgb[58:63, 104:115] = (30, 190, 60)  # attached leaf/hand, 55 pixels
        background = module.estimate_background(self.rgb)
        self.assertFalse(background['confident'])

        def drop_accent(cell, session):
            result = np.array(cell.convert('RGBA'))
            result[58:63, 104:115, 3] = 0
            return Image.fromarray(result)

        remove = Mock(side_effect=drop_accent)
        session = Mock(return_value=object())
        rembg = SimpleNamespace(new_session=session, remove=remove)
        with patch.dict(sys.modules, {'rembg': rembg}):
            out, report = self.run_cli(self.rgb)
            codes = {item['code'] for item in report['cells'][0]['findings']}
            self.assertEqual(report['effective_mode'], 'rembg')
            self.assertEqual(report['review_cells'], [1])
            self.assertIn('uncertain_background', codes)
            self.assertIn('candidate_disagreement', codes)
            self.assertNotIn('possible_cutout_loss', codes)
            disagreement = next(item for item in report['cells'][0]['findings'] if item['code'] == 'candidate_disagreement')
            self.assertEqual(disagreement['pixels'], 55)
            with Image.open(out / report['cells'][0]['colour_candidate']) as image:
                candidate = np.asarray(image)
                self.assertEqual(image.size, (128, 128))
            self.assertTrue(np.all(candidate[58:63, 104:115, 3] == 255))
            self.assertTrue((out / 'qa/colour-candidate.png').is_file())
            remove.reset_mock()
            session.reset_mock()
            _, corrected = self.run_cli(self.rgb, repairs={'cells': {'1': {'mode': 'floodfill'}}})
            self.assertEqual(corrected['effective_mode'], 'floodfill')
            self.assertNotIn('colour_candidate', corrected['cells'][0])
            self.assertFalse((out / 'qa/colour-candidate.png').exists())
            self.assertFalse((out / 'qa/colour-candidates/01.png').exists())
            remove.assert_not_called()
            session.assert_not_called()

    def test_cell_override_supports_mixed_modes_and_local_tolerance(self):
        rgb = np.full((256, 256, 3), self.bg, dtype=np.uint8)
        for y in (24, 152):
            for x in (24, 152):
                rgb[y:y+64, x:x+64] = (240, 210, 160)
        remove = Mock(side_effect=lambda cell, session: cell.convert('RGBA'))
        rembg = SimpleNamespace(new_session=Mock(return_value=object()), remove=remove)
        with patch.dict(sys.modules, {'rembg': rembg}):
            out, report = self.run_cli(rgb, '--grid', '2', '--background-tolerance', '12', repairs={
                'cells': {'2': {'mode': 'rembg'}, '3': {'background_tolerance': 32}},
            })
        self.assertEqual(report['effective_mode'], 'mixed')
        self.assertEqual([entry['mode'] for entry in report['cells']], ['floodfill', 'rembg', 'floodfill', 'floodfill'])
        self.assertEqual([entry['background_tolerance'] for entry in report['cells']], [12, 12, 32, 12])
        self.assertEqual(sorted(path.name for path in (out / 'qa/colour-candidates').glob('*.png')), ['02.png'])
        remove.assert_called_once()


if __name__ == '__main__':
    unittest.main()
