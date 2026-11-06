import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('slice_icons', Path(__file__).parents[1] / 'scripts/slice_icons.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bg = np.array([136,136,135])
        self.rgb = np.full((128,128,3), self.bg, dtype=np.uint8)
        self.rgb[24:104,24:104] = [240,210,160]
        self.rgb[44:64,44:64] = self.bg

    def run_cli(self, *extra, rgb=None):
        sheet = self.root / 'sheet.png'
        Image.fromarray(self.rgb if rgb is None else rgb).save(sheet)
        return self.slice_existing(sheet, *extra)

    def slice_existing(self, sheet, *extra, output=None):
        out = output if output is not None else self.root / 'out'
        argv = ['slice_icons', str(sheet), str(out), '--grid', '1', '--size', '128', *extra]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        return json.loads((out / 'qa/report.json').read_text())

    def review_file(self, report, status='accepted', decisions=None):
        entry = report['cells'][0]
        path = self.root / 'review.json'
        path.write_text(json.dumps({'cells': {'1': {
            'fingerprint': entry['fingerprint'], 'status': status, 'reviewer': 'test reviewer',
            'note': 'Compared source and final output at original size on all three backgrounds.',
            'decisions': {'transparent_holes': 'valid_hole'} if decisions is None else decisions,
        }}}))
        return path

    def test_acceptance_preserves_findings_and_png_but_changes_preview_and_status(self):
        first = self.run_cli()
        original = (self.root / 'out/01.png').read_bytes()
        preview = (self.root / 'out/qa/magenta.png').read_bytes()
        path = self.review_file(first)
        accepted = self.run_cli('--review-file', str(path))
        self.assertEqual(accepted['status'], 'accepted')
        self.assertEqual(accepted['accepted_cells'], [1])
        self.assertFalse(accepted['review_cells'])
        self.assertFalse(accepted['manual_review_required'])
        self.assertEqual(first['cells'][0]['findings'], accepted['cells'][0]['findings'])
        self.assertEqual(original, (self.root / 'out/01.png').read_bytes())
        self.assertNotEqual(preview, (self.root / 'out/qa/magenta.png').read_bytes())
        self.assertEqual(accepted['cells'][0]['output_sha256'], hashlib.sha256(original).hexdigest())
        summary = (self.root / 'out/qa/summary.md').read_text()
        self.assertIn('valid_hole', summary)
        self.assertIn('accepted 1/1', summary)

    def test_source_and_parameter_changes_invalidate_acceptance(self):
        first = self.run_cli()
        path = self.review_file(first)
        self.rgb[30,30] = [20,80,200]
        changed = self.run_cli('--review-file', str(path))
        self.assertEqual(changed['cells'][0]['review']['status'], 'stale')
        self.assertTrue(changed['manual_review_required'])
        path = self.review_file(changed)
        resized = self.run_cli('--size', '256', '--review-file', str(path))
        self.assertEqual(resized['cells'][0]['review']['status'], 'stale')

    def test_repair_changes_invalidate_only_the_affected_cell(self):
        rgb = np.full((256,256,3), self.bg, dtype=np.uint8)
        for y in (24,152):
            for x in (24,152):
                rgb[y:y+80,x:x+80] = [240,210,160]
                rgb[y+20:y+40,x+20:x+40] = self.bg
        first = self.run_cli('--grid', '2', rgb=rgb)
        path = self.root / 'review.json'
        path.write_text(json.dumps({'cells': {str(cell['cell']): {
            'fingerprint': cell['fingerprint'], 'status': 'accepted', 'reviewer': 'test reviewer',
            'note': 'Compared this icon with its source and final output.',
            'decisions': {'transparent_holes': 'valid_hole'},
        } for cell in first['cells']}}))
        accepted = self.run_cli('--grid', '2', '--review-file', str(path), rgb=rgb)
        self.assertEqual(accepted['status'], 'accepted')
        repairs = self.root / 'repairs.json'
        repairs.write_text(json.dumps({'cells': {'1': {'background_tolerance': 12}}}))
        changed = self.run_cli('--grid', '2', '--review-file', str(path), '--repairs', str(repairs), rgb=rgb)
        self.assertEqual(changed['review_cells'], [1])
        self.assertEqual(changed['accepted_cells'], [2,3,4])
        self.assertEqual(changed['cells'][0]['review']['status'], 'stale')
        self.assertEqual(changed['cells'][0]['review']['fingerprint'], first['cells'][0]['fingerprint'])

    def test_output_change_invalidates_acceptance_even_with_same_source_and_flags(self):
        first = self.run_cli()
        path = self.review_file(first)
        original = module.normalise
        def modified(*args):
            image = original(*args)
            image.putpixel((60,60), (1,2,3,255))
            return image
        with patch.object(module, 'normalise', modified):
            changed = self.run_cli('--review-file', str(path))
        self.assertEqual(changed['cells'][0]['review']['status'], 'stale')

    def test_missing_decisions_remain_pending_and_explicit_failures_block(self):
        first = self.run_cli()
        path = self.review_file(first, decisions={})
        pending = self.run_cli('--review-file', str(path))
        self.assertEqual(pending['cells'][0]['review']['status'], 'pending')
        path = self.review_file(first, status='needs_fix', decisions={'transparent_holes':'needs_fix'})
        failed = self.run_cli('--review-file', str(path))
        self.assertEqual(failed['status'], 'needs_fix')
        self.assertEqual(failed['needs_fix_cells'], [1])

    def test_empty_export_cannot_be_accepted(self):
        self.rgb[:] = self.bg
        first = self.run_cli()
        path = self.review_file(first, decisions={'empty':'false_positive'})
        failed = self.run_cli('--review-file', str(path))
        self.assertEqual(failed['status'], 'needs_fix')

    def test_resized_empty_export_is_an_error_and_cannot_be_accepted(self):
        rgb = np.full((512,512,3), self.bg, dtype=np.uint8)
        for y in (64,256,448):
            for x in (64,256,448):
                rgb[y,x] = [240,210,160]
        first = self.run_cli('--size', '16', rgb=rgb)
        with Image.open(self.root / 'out/01.png') as image:
            self.assertIsNone(image.getchannel('A').getbbox())
        self.assertEqual(first['cells'][0]['retained_components'], 9)
        self.assertEqual(first['cells'][0]['automatic_status'], 'error')
        self.assertEqual(first['status'], 'needs_fix')
        self.assertEqual(sum(f['code'] == 'empty' for f in first['cells'][0]['findings']), 1)
        path = self.review_file(first, decisions={'empty': 'false_positive'})
        result = self.run_cli('--size', '16', '--review-file', str(path), rgb=rgb)
        self.assertEqual(result['status'], 'needs_fix')
        self.assertFalse(result['accepted_cells'])
        self.assertTrue(result['manual_review_required'])

    def accepted_grid(self):
        rgb = np.full((256,256,3), self.bg, dtype=np.uint8)
        for y in (24,152):
            for x in (24,152):
                rgb[y:y+80,x:x+80] = [240,210,160]
        first = self.run_cli('--grid', '2', rgb=rgb)
        path = self.root / 'review.json'
        path.write_text(json.dumps({'cells': {str(cell['cell']): {
            'fingerprint': cell['fingerprint'], 'status': 'accepted', 'reviewer': 'test reviewer',
            'note': 'Compared each source and final output.', 'decisions': {},
        } for cell in first['cells']}}))
        result = self.run_cli('--grid', '2', '--review-file', str(path), rgb=rgb)
        self.assertEqual(result['status'], 'accepted')
        return rgb, path

    def output_snapshot(self):
        out = self.root / 'out'
        return {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*') if p.is_file()}

    def test_interrupted_rerun_preserves_the_complete_accepted_batch(self):
        rgb, path = self.accepted_grid()
        original = self.output_snapshot()
        rgb[24:104,24:104] = [30,190,60]
        normalise = module.normalise
        for error in (RuntimeError('cell failed'), KeyboardInterrupt()):
            calls = 0
            def interrupt_second_cell(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise error
                return normalise(*args)
            with self.subTest(error=type(error).__name__), patch.object(module, 'normalise', interrupt_second_cell):
                with self.assertRaises(type(error)):
                    self.run_cli('--grid', '2', '--review-file', str(path), rgb=rgb)
            self.assertEqual(self.output_snapshot(), original)
            self.assertFalse(list(self.root.glob('.out-*')))

    def test_qa_write_failure_preserves_the_complete_accepted_batch(self):
        rgb, path = self.accepted_grid()
        original = self.output_snapshot()
        rgb[24:104,24:104] = [30,190,60]
        with patch.object(module, 'save_qa_report', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.run_cli('--grid', '2', '--review-file', str(path), rgb=rgb)
        self.assertEqual(self.output_snapshot(), original)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_publication_failure_rolls_back_before_restoring_acceptance(self):
        rgb, path = self.accepted_grid()
        (self.root / 'out/qa/reviewer-note.txt').write_text('Keep this note.')
        (self.root / 'out/readme.txt').write_text('Keep this file.')
        original = self.output_snapshot()
        rgb[24:104,24:104] = [30,190,60]
        replace = Path.replace
        failed = False
        def fail_new_qa(source, target):
            nonlocal failed
            target = Path(target)
            if (source.name == 'qa' and '-stage-' in str(source.parent)
                    and target == (self.root / 'out/qa').resolve()):
                failed = True
                self.assertNotEqual((self.root / 'out/01.png').read_bytes(), original['01.png'])
                self.assertFalse((self.root / 'out/qa/report.json').exists())
                raise OSError('publication interrupted')
            return replace(source, target)
        with patch.object(Path, 'replace', fail_new_qa), self.assertRaises(OSError):
            self.run_cli('--grid', '2', '--review-file', str(path), rgb=rgb)
        self.assertTrue(failed)
        self.assertEqual(self.output_snapshot(), original)
        self.assertFalse(list(self.root.glob('.out-*')))
        result = self.run_cli('--grid', '2', '--review-file', str(path), rgb=rgb)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual((self.root / 'out/qa/reviewer-note.txt').read_text(), 'Keep this note.')
        self.assertEqual((self.root / 'out/readme.txt').read_text(), 'Keep this file.')

    def crash_publication(self, point, *extra, sheet=None, recover_only=False, output=None):
        # os._exit bypasses exceptions, finally blocks and TemporaryDirectory
        # cleanup, matching an actual killed process rather than an exception.
        child = r"""
import importlib.util
import os
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('slice_icons', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
point, recover_only, sheet, output = sys.argv[2:6]
output = Path(output).resolve()
replace = Path.replace

def crash_after_move(source, target):
    target = Path(target)
    result = replace(source, target)
    event = None
    if source.parent == output and '-backup-' in target.parent.name:
        event = 'backup_' + source.name
    elif '-stage-' in source.parent.name and target.parent == output:
        event = 'install_' + source.name
    elif '-backup-' in source.parent.name and target.parent == output:
        event = 'restore_' + source.name
    elif target.parent.name == 'discarded':
        event = 'discard_' + source.name
    elif source.name == 'transaction.next.json':
        import json
        event = 'journal_' + json.loads(target.read_text())['state']
    if event == point:
        os._exit(97)
    return result

Path.replace = crash_after_move
if point in ('cleanup_backup', 'cleanup_backup_qa'):
    import shutil
    rmtree = shutil.rmtree
    def crash_during_cleanup(path, *args, **kwargs):
        path = Path(path)
        if '-backup-' in path.name:
            partial = path / ('qa/report.json' if point == 'cleanup_backup_qa' else '01.png')
            partial.unlink()
            os._exit(97)
        return rmtree(path, *args, **kwargs)
    shutil.rmtree = crash_during_cleanup
if recover_only == 'True':
    module.recover_publication(output)
else:
    sys.argv = ['slice_icons', sheet, str(output), '--grid', '1', '--size', '128', *sys.argv[6:]]
    module.main()
"""
        result = subprocess.run(
            [sys.executable, '-c', child, str(Path(module.__file__).resolve()), point,
             str(recover_only), str(sheet or self.root / 'sheet.png'), str(output or self.root / 'out'), *extra],
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 97, result.stdout + result.stderr)

    def test_forced_exit_at_each_publication_boundary_restores_previous_batch(self):
        rgb, _ = self.accepted_grid()
        out = self.root / 'out'
        (out / 'qa/reviewer-note.txt').write_text('Keep this note.')
        (out / '99.png').write_bytes(b'untracked asset')
        original = self.output_snapshot()
        rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(rgb).save(self.root / 'sheet.png')
        for point in ('journal_pending', 'backup_qa', 'backup_01.png',
                      'backup_04.png', 'install_01.png', 'install_04.png', 'install_qa'):
            with self.subTest(point=point):
                self.crash_publication(point, '--grid', '2')
                self.assertTrue(module.transaction_path(out).is_file())
                module.recover_publication(out)
                self.assertEqual(self.output_snapshot(), original)
                self.assertFalse(list(self.root.glob('.out-*')))

    def test_rerun_after_forced_exit_recovers_before_processing(self):
        rgb, path = self.accepted_grid()
        original = self.output_snapshot()
        rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(rgb).save(self.root / 'sheet.png')
        self.crash_publication('install_01.png', '--grid', '2')
        self.assertFalse((self.root / 'out/qa/report.json').exists())
        report = self.slice_existing(self.root / 'sheet.png', '--grid', '2', '--review-file', str(path))
        self.assertEqual(report['status'], 'needs_review')
        self.assertNotEqual((self.root / 'out/01.png').read_bytes(), original['01.png'])
        for cell in report['cells']:
            self.assertEqual(cell['output_sha256'], hashlib.sha256((self.root / 'out' / cell['file']).read_bytes()).hexdigest())
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_first_publication_crash_and_smaller_rerun_preserve_untracked_files(self):
        out = self.root / 'out'
        out.mkdir()
        (out / '99.png').write_bytes(b'untracked asset')
        Image.fromarray(self.rgb).save(self.root / 'sheet.png')
        self.crash_publication('install_01.png')
        report = self.slice_existing(self.root / 'sheet.png')
        self.assertEqual(len(report['cells']), 1)
        self.assertEqual((out / '99.png').read_bytes(), b'untracked asset')
        self.accepted_grid()
        self.crash_publication('install_01.png', '--grid', '2')
        report = self.slice_existing(self.root / 'sheet.png')
        self.assertEqual(sorted(p.name for p in out.glob('*.png')), ['01.png', '99.png'])
        self.assertEqual(len(report['cells']), 1)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_recovery_is_repeatable_after_it_is_killed(self):
        rgb, _ = self.accepted_grid()
        original = self.output_snapshot()
        rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(rgb).save(self.root / 'sheet.png')
        for point in ('discard_qa', 'discard_01.png', 'restore_01.png', 'restore_qa', 'journal_rolled_back'):
            with self.subTest(point=point):
                self.crash_publication('install_qa', '--grid', '2')
                self.crash_publication(point, recover_only=True)
                module.recover_publication(self.root / 'out')
                self.assertEqual(self.output_snapshot(), original)
                self.assertFalse(list(self.root.glob('.out-*')))

    def test_committed_publication_is_not_rolled_back_after_exit(self):
        rgb, _ = self.accepted_grid()
        rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(rgb).save(self.root / 'sheet.png')
        self.crash_publication('journal_committed', '--grid', '2')
        complete = self.output_snapshot()
        module.recover_publication(self.root / 'out')
        self.assertEqual(self.output_snapshot(), complete)
        self.assertEqual(json.loads(complete['qa/report.json'])['status'], 'needs_review')
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_modified_partial_output_blocks_recovery_without_touching_files(self):
        self.accepted_grid()
        self.crash_publication('install_01.png', '--grid', '2')
        (self.root / 'out/01.png').write_bytes(b'changed after crash')
        partial = self.output_snapshot()
        with self.assertRaisesRegex(ValueError, 'modified recovery file'):
            self.slice_existing(self.root / 'sheet.png', '--grid', '2')
        self.assertEqual(self.output_snapshot(), partial)
        self.assertTrue(module.transaction_path(self.root / 'out').is_file())

    def test_modified_backup_blocks_recovery_before_any_mutation(self):
        self.accepted_grid()
        self.crash_publication('install_01.png', '--grid', '2')
        backup = next(self.root.glob('.out-backup-*'))
        (backup / 'qa/report.json').write_text('changed after crash')
        partial = self.output_snapshot()
        with self.assertRaisesRegex(ValueError, 'Modified publication backup'):
            module.recover_publication(self.root / 'out')
        self.assertEqual(self.output_snapshot(), partial)
        self.assertEqual((backup / 'qa/report.json').read_text(), 'changed after crash')

    def test_partial_output_selected_as_new_input_is_not_removed(self):
        rgb, _ = self.accepted_grid()
        rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(rgb).save(self.root / 'sheet.png')
        self.crash_publication('install_01.png', '--grid', '2')
        partial = self.output_snapshot()
        with self.assertRaisesRegex(FileExistsError, 'input'):
            self.slice_existing(self.root / 'out/01.png')
        self.assertEqual(self.output_snapshot(), partial)

    def test_missing_input_under_qa_is_restored_before_opening(self):
        self.run_cli()
        source = self.root / 'out/qa/personal-source.png'
        Image.fromarray(self.rgb).save(source)
        for point in ('install_01.png', 'install_qa'):
            with self.subTest(point=point):
                self.crash_publication(point, sheet=source)
                self.assertEqual(source.exists(), point == 'install_qa')
                report = self.slice_existing(source)
                self.assertTrue(source.is_file())
                self.assertEqual(report['sheet'], str(source.resolve()))
                self.assertFalse(list(self.root.glob('.out-*')))

    def test_input_in_transaction_directory_survives_pending_and_committed_recovery(self):
        for point in ('install_01.png', 'journal_committed'):
            with self.subTest(point=point):
                self.accepted_grid()
                self.crash_publication(point, '--grid', '2')
                backup = next(self.root.glob('.out-backup-*'))
                source = backup / '01.png'
                original = source.read_bytes()
                live = self.output_snapshot()
                with self.assertRaisesRegex(FileExistsError, 'input'):
                    self.slice_existing(source)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(self.output_snapshot(), live)
                module.recover_publication(self.root / 'out')

    def test_cleanup_interruption_keeps_committed_outputs_and_can_resume(self):
        self.accepted_grid()
        self.crash_publication('cleanup_backup', '--grid', '2')
        complete = self.output_snapshot()
        module.recover_publication(self.root / 'out')
        self.assertEqual(self.output_snapshot(), complete)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_recovery_quarantine_symlink_cannot_move_outputs_outside_backup(self):
        self.accepted_grid()
        self.crash_publication('install_01.png', '--grid', '2')
        backup = next(self.root.glob('.out-backup-*'))
        outside = self.root / 'outside'
        outside.mkdir()
        (backup / 'discarded').symlink_to(outside, target_is_directory=True)
        live = self.output_snapshot()
        with self.assertRaisesRegex(ValueError, 'Unsafe recovery quarantine'):
            module.recover_publication(self.root / 'out')
        self.assertEqual(self.output_snapshot(), live)
        self.assertFalse(list(outside.iterdir()))

    def transaction_snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for root in self.root.glob('.out-*')
                for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file()}

    def test_quarantined_edits_block_recovery_before_any_file_changes(self):
        self.accepted_grid()
        self.crash_publication('install_qa', '--grid', '2')
        self.crash_publication('discard_qa', recover_only=True)
        backup = next(self.root.glob('.out-backup-*'))
        original_report = (backup / 'discarded/qa/report.json').read_bytes()
        for name, contents in [('reviewer-note.txt', b'New review after interruption.'),
                               ('report.json', b'Changed report after interruption.')]:
            with self.subTest(name=name):
                edited = backup / 'discarded/qa' / name
                edited.write_bytes(contents)
                live, evidence = self.output_snapshot(), self.transaction_snapshot()
                with self.assertRaisesRegex(ValueError, 'Modified or untracked recovery file'):
                    module.recover_publication(self.root / 'out')
                self.assertEqual(self.output_snapshot(), live)
                self.assertEqual(self.transaction_snapshot(), evidence)
                if name == 'report.json':
                    edited.write_bytes(original_report)
                else:
                    edited.unlink()
        module.recover_publication(self.root / 'out')
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_terminal_cleanup_checks_changed_backup_and_new_stage_files(self):
        self.accepted_grid()
        self.crash_publication('journal_committed', '--grid', '2')
        backup = next(self.root.glob('.out-backup-*'))
        stage = next(self.root.glob('.out-stage-*'))
        old = (backup / '01.png').read_bytes()
        for edited, contents in ((backup / '01.png', b'Edited old PNG'),
                                 (stage / 'new-note.txt', b'Note added after interruption')):
            with self.subTest(path=edited.name):
                edited.write_bytes(contents)
                live, evidence = self.output_snapshot(), self.transaction_snapshot()
                with self.assertRaisesRegex(ValueError, 'Modified or untracked recovery file'):
                    module.recover_publication(self.root / 'out')
                self.assertEqual(self.output_snapshot(), live)
                self.assertEqual(self.transaction_snapshot(), evidence)
                if edited == backup / '01.png':
                    edited.write_bytes(old)
                else:
                    edited.unlink()
        module.recover_publication(self.root / 'out')
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_terminal_cleanup_resumes_after_part_of_qa_was_deleted(self):
        self.accepted_grid()
        self.crash_publication('cleanup_backup_qa', '--grid', '2')
        complete = self.output_snapshot()
        backup = next(self.root.glob('.out-backup-*'))
        self.assertFalse((backup / 'qa/report.json').exists())
        self.assertTrue((backup / 'qa').is_dir())
        module.recover_publication(self.root / 'out')
        self.assertEqual(self.output_snapshot(), complete)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_legacy_transactions_recover_with_verified_complete_trees(self):
        self.accepted_grid()
        original = self.output_snapshot()
        self.crash_publication('install_qa', '--grid', '2')
        self.crash_publication('discard_qa', recover_only=True)
        journal = module.transaction_path(self.root / 'out')
        record = json.loads(journal.read_text())
        record['version'] = 1
        del record['cleanup']
        journal.write_text(json.dumps(record))
        backup = next(self.root.glob('.out-backup-*'))
        note = backup / 'discarded/qa/note.txt'
        note.write_text('Keep this note.')
        evidence = self.transaction_snapshot()
        with self.assertRaisesRegex(ValueError, 'Modified or untracked recovery file'):
            module.recover_publication(self.root / 'out')
        self.assertEqual(self.transaction_snapshot(), evidence)
        note.unlink()
        module.recover_publication(self.root / 'out')
        self.assertEqual(self.output_snapshot(), original)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_qa_edits_during_slicing_are_preserved_without_publication(self):
        self.run_cli()
        out = self.root / 'out'
        note = out / 'qa/reviewer-note.md'
        original_normalise = module.normalise
        for change in ('edit', 'add', 'delete'):
            with self.subTest(change=change):
                if change == 'add':
                    note.unlink(missing_ok=True)
                else:
                    note.write_text('Original note')
                expected = None
                def edit_while_slicing(*args, **kwargs):
                    nonlocal expected
                    if change == 'delete':
                        note.unlink()
                    else:
                        note.write_text('New review finding during slicing')
                    expected = module.publication_fingerprint(out)
                    return original_normalise(*args, **kwargs)
                with patch.object(module, 'normalise', edit_while_slicing), self.assertRaisesRegex(FileExistsError, 'QA changed'):
                    self.slice_existing(self.root / 'sheet.png')
                self.assertEqual(module.publication_fingerprint(out), expected)
                self.assertFalse(list(self.root.glob('.out-*')))
        # Once the edit has finished, retrying preserves it and can publish.
        note.write_text('Finished review note')
        self.slice_existing(self.root / 'sheet.png')
        self.assertEqual(note.read_text(), 'Finished review note')

    def test_qa_copy_detects_concurrent_note_edits(self):
        self.run_cli()
        out = self.root / 'out'
        note = out / 'qa/reviewer-note.md'
        note.write_text('Original note')
        copytree = module.shutil.copytree
        expected = None
        def copy_then_edit(source, target, *args, **kwargs):
            nonlocal expected
            result = copytree(source, target, *args, **kwargs)
            if Path(source) == (out / 'qa').resolve():
                note.write_text('Edited while copying')
                expected = module.publication_fingerprint(out)
            return result
        with patch.object(module.shutil, 'copytree', copy_then_edit), self.assertRaisesRegex(FileExistsError, 'QA changed while copying'):
            self.slice_existing(self.root / 'sheet.png')
        self.assertEqual(module.publication_fingerprint(out), expected)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_qa_created_during_first_slice_is_not_overwritten(self):
        sheet = self.root / 'sheet.png'
        Image.fromarray(self.rgb).save(sheet)
        out = self.root / 'out'
        normalise = module.normalise
        def create_qa(*args, **kwargs):
            (out / 'qa').mkdir(parents=True)
            (out / 'qa/note.md').write_text('New note')
            return normalise(*args, **kwargs)
        with patch.object(module, 'normalise', create_qa), self.assertRaisesRegex(FileExistsError, 'QA changed'):
            self.slice_existing(sheet)
        self.assertEqual((out / 'qa/note.md').read_text(), 'New note')
        self.assertFalse((out / '01.png').exists())
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_qa_edit_at_publication_is_kept_in_backup(self):
        self.run_cli()
        out = (self.root / 'out').resolve()
        note = out / 'qa/note.md'
        note.write_text('Original note')
        original_png = (out / '01.png').read_bytes()
        replace = Path.replace
        def edit_then_move(source, target):
            if source == out / 'qa':
                note.write_text('Edited immediately before rename')
            return replace(source, target)
        with patch.object(Path, 'replace', edit_then_move), self.assertRaisesRegex(ValueError, 'Modified publication backup'):
            self.slice_existing(self.root / 'sheet.png')
        backup = next(self.root.glob('.out-backup-*'))
        self.assertEqual((backup / 'qa/note.md').read_text(), 'Edited immediately before rename')
        self.assertEqual((out / '01.png').read_bytes(), original_png)
        self.assertTrue(module.transaction_path(out).exists())
        self.assertFalse((out / 'qa/report.json').exists())

    def test_user_qa_links_keep_their_targets_and_dangling_links(self):
        self.run_cli()
        qa = self.root / 'out/qa'
        external = self.root / 'notes'
        external.mkdir()
        note = external / 'note.md'
        note.write_text('Original external note')
        links = {
            'reviewer-note.md': Path('../../notes/note.md'),
            'absolute-note.md': note,
            'reference': Path('../../notes'),
            'missing.md': Path('../../archived-note.md'),
            'cycle-a': Path('cycle-b'),
            'cycle-b': Path('cycle-a'),
        }
        for name, target in links.items():
            (qa / name).symlink_to(target, target_is_directory=name == 'reference')
        report = self.slice_existing(self.root / 'sheet.png')
        self.assertEqual(report['exported'], 1)
        for name, target in links.items():
            self.assertTrue((qa / name).is_symlink(), name)
            self.assertEqual((qa / name).readlink(), target)
        note.write_text('Updated external note')
        self.assertEqual((qa / 'reviewer-note.md').read_text(), 'Updated external note')
        self.assertEqual((qa / 'absolute-note.md').read_text(), 'Updated external note')
        self.assertEqual((qa / 'reference/note.md').read_text(), 'Updated external note')
        self.assertFalse((qa / 'missing.md').exists())

    def test_generated_qa_links_cannot_modify_external_targets(self):
        # Exercise existing and missing targets, directories, stale entries,
        # and uppercase numeric aliases on case-insensitive filesystems.
        for relative in ('qa', 'qa/details', 'qa/colour-candidates',
                         'qa/report.json', 'qa/summary.md', 'qa/source.png',
                         'qa/colour-candidate.png', 'qa/details/01.json',
                         'qa/details/99.json', 'qa/details/01.JSON',
                         'qa/colour-candidates/01.png', 'qa/colour-candidates/99.png'):
            for dangling in (False, True):
                with self.subTest(relative=relative, dangling=dangling), tempfile.TemporaryDirectory(dir=self.root) as directory:
                    root = Path(directory)
                    out = root / 'out'
                    link = out / relative
                    link.parent.mkdir(parents=True)
                    target = root / 'external'
                    is_directory = relative in ('qa', 'qa/details', 'qa/colour-candidates')
                    if not dangling:
                        if is_directory:
                            target.mkdir()
                            (target / 'note.txt').write_text('External content')
                        else:
                            target.write_text('External content')
                    link.symlink_to(target, target_is_directory=is_directory)
                    before = module.publication_fingerprint(root)
                    sheet = self.root / 'sheet.png'
                    Image.fromarray(self.rgb).save(sheet)
                    with self.assertRaisesRegex(FileExistsError, 'symlink'):
                        self.slice_existing(sheet, output=out)
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(link.readlink(), target)
                    # The reusable lock is the only new file permitted.
                    (root / '.out.lock').unlink()
                    self.assertEqual(module.publication_fingerprint(root), before)

    def test_preserved_qa_links_survive_failed_publication(self):
        self.run_cli()
        out = (self.root / 'out').resolve()
        qa = out / 'qa'
        (qa / 'dangling.md').symlink_to('../../not-yet-written.md')
        (qa / 'source-reference.png').symlink_to('../../sheet.png')
        before = module.publication_fingerprint(out)
        replace = Path.replace
        def fail_qa(source, target):
            if source.name == 'qa' and '-stage-' in source.parent.name:
                raise OSError('Interrupted publication')
            return replace(source, target)
        with patch.object(Path, 'replace', fail_qa), self.assertRaisesRegex(OSError, 'Interrupted publication'):
            self.slice_existing(self.root / 'sheet.png')
        self.assertEqual(module.publication_fingerprint(out), before)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_preserved_qa_links_survive_forced_exit_recovery(self):
        self.run_cli()
        out = self.root / 'out'
        (out / 'qa/dangling.md').symlink_to('../../not-yet-written.md')
        (out / 'qa/source-reference.png').symlink_to('../../sheet.png')
        before = module.publication_fingerprint(out)
        self.crash_publication('install_qa')
        module.recover_publication(out)
        self.assertEqual(module.publication_fingerprint(out), before)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_qa_source_symlink_is_not_replaced_with_generated_preview(self):
        self.run_cli()
        source = self.root / 'out/qa/source.png'
        source.unlink()
        source.symlink_to(self.root / 'sheet.png')
        original = self.output_snapshot()
        external = (self.root / 'sheet.png').read_bytes()
        with self.assertRaisesRegex(FileExistsError, 'input'):
            self.slice_existing(source)
        self.assertTrue(source.is_symlink())
        self.assertEqual(self.output_snapshot(), original)
        self.assertEqual((self.root / 'sheet.png').read_bytes(), external)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_qa_review_and_repair_input_symlinks_are_protected(self):
        first = self.run_cli()
        review = self.review_file(first)
        repairs = self.root / 'repairs.json'
        repairs.write_text(json.dumps({'cells': {}}))
        for option, target in (('--review-file', review), ('--repairs', repairs)):
            with self.subTest(option=option):
                link = self.root / 'out/qa/input.json'
                link.symlink_to(target)
                expected_target, expected_bytes = link.readlink(), target.read_bytes()
                report = self.slice_existing(self.root / 'sheet.png', option, str(link))
                self.assertEqual(report['exported'], 1)
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.readlink(), expected_target)
                self.assertEqual(target.read_bytes(), expected_bytes)
                link.unlink()

    def test_input_through_qa_directory_symlink_is_protected(self):
        self.run_cli()
        directory = self.root / 'out/qa/reference'
        external = self.root / 'external'
        external.mkdir()
        Image.fromarray(self.rgb).save(external / 'input.png')
        directory.symlink_to(external, target_is_directory=True)
        expected_target = directory.readlink()
        report = self.slice_existing(directory / 'input.png')
        self.assertEqual(report['exported'], 1)
        self.assertTrue(directory.is_symlink())
        self.assertEqual(directory.readlink(), expected_target)
        self.assertEqual((directory / 'input.png').read_bytes(), (external / 'input.png').read_bytes())

    def test_multihop_source_links_are_not_replaced_with_generated_preview(self):
        self.run_cli()
        intermediate = self.root / 'out/qa/source.png'
        intermediate.unlink()
        intermediate.symlink_to('../../sheet.png')
        supplied = self.root / 'input.png'
        for target in (Path('out/qa/source.png'), intermediate.resolve().parent / 'out/qa/source.png'):
            with self.subTest(target=target):
                supplied.symlink_to(target)
                original = self.output_snapshot()
                source_bytes = supplied.read_bytes()
                with self.assertRaisesRegex(FileExistsError, 'input'):
                    self.slice_existing(supplied)
                self.assertTrue(intermediate.is_symlink())
                self.assertTrue(supplied.is_symlink())
                self.assertEqual(supplied.read_bytes(), source_bytes)
                self.assertEqual(self.output_snapshot(), original)
                self.assertFalse(list(self.root.glob('.out-*')))
                supplied.unlink()

    def test_multihop_review_and_repair_links_are_protected(self):
        first = self.run_cli()
        review = self.review_file(first)
        repairs = self.root / 'repairs.json'
        repairs.write_text(json.dumps({'cells': {}}))
        supplied = self.root / 'input.json'
        intermediate = self.root / 'out/qa/input.json'
        for option, target in (('--review-file', review), ('--repairs', repairs)):
            with self.subTest(option=option):
                intermediate.symlink_to(target)
                supplied.symlink_to('out/qa/input.json')
                source_bytes = supplied.read_bytes()
                expected_target = intermediate.readlink()
                report = self.slice_existing(self.root / 'sheet.png', option, str(supplied))
                self.assertEqual(report['exported'], 1)
                self.assertTrue(intermediate.is_symlink())
                self.assertEqual(intermediate.readlink(), expected_target)
                self.assertTrue(supplied.is_symlink())
                self.assertEqual(supplied.read_bytes(), source_bytes)
                supplied.unlink()
                intermediate.unlink()

    def test_file_link_target_through_directory_link_is_protected(self):
        self.run_cli()
        external = self.root / 'external'
        external.mkdir()
        Image.fromarray(self.rgb).save(external / 'input.png')
        intermediate = self.root / 'out/qa/reference'
        intermediate.symlink_to('../../external', target_is_directory=True)
        supplied = self.root / 'input.png'
        supplied.symlink_to('out/qa/reference/input.png')
        original = (external / 'input.png').read_bytes()
        report = self.slice_existing(supplied)
        self.assertEqual(report['exported'], 1)
        self.assertTrue(intermediate.is_symlink())
        self.assertEqual(intermediate.readlink(), Path('../../external'))
        self.assertTrue(supplied.is_symlink())
        self.assertEqual(supplied.read_bytes(), original)

    def test_multihop_directory_links_are_protected(self):
        self.run_cli()
        external = self.root / 'external'
        external.mkdir()
        Image.fromarray(self.rgb).save(external / 'input.png')
        intermediate = self.root / 'out/qa/reference'
        intermediate.symlink_to('../../external', target_is_directory=True)
        directory = self.root / 'reference'
        directory.symlink_to('out/qa/reference', target_is_directory=True)
        original = (external / 'input.png').read_bytes()
        report = self.slice_existing(directory / 'input.png')
        self.assertEqual(report['exported'], 1)
        self.assertTrue(intermediate.is_symlink())
        self.assertEqual(intermediate.readlink(), Path('../../external'))
        self.assertTrue(directory.is_symlink())
        self.assertEqual((directory / 'input.png').read_bytes(), original)

    def test_multihop_backup_input_blocks_pending_and_committed_cleanup(self):
        for point in ('install_01.png', 'journal_committed'):
            with self.subTest(point=point):
                self.run_cli()
                original_link = self.root / 'out/qa/original-link.png'
                original_link.unlink(missing_ok=True)
                original_link.symlink_to(self.root / 'sheet.png')
                self.crash_publication(point)
                backup = next(self.root.glob('.out-backup-*'))
                intermediate = backup / 'qa/original-link.png'
                supplied = self.root / 'input.png'
                supplied.symlink_to(intermediate)
                source_bytes = supplied.read_bytes()
                live, evidence = self.output_snapshot(), self.transaction_snapshot()
                with self.assertRaisesRegex(FileExistsError, 'input'):
                    self.slice_existing(supplied)
                self.assertTrue(intermediate.is_symlink())
                self.assertEqual(supplied.read_bytes(), source_bytes)
                self.assertEqual(self.output_snapshot(), live)
                self.assertEqual(self.transaction_snapshot(), evidence)
                supplied.unlink()
                module.recover_publication(self.root / 'out')
                self.assertFalse(list(self.root.glob('.out-*')))

    def test_cyclic_input_links_block_recovery_without_mutation(self):
        self.run_cli()
        self.crash_publication('install_01.png')
        supplied, other = self.root / 'input.png', self.root / 'other.png'
        supplied.symlink_to(other.name)
        other.symlink_to(supplied.name)
        live, evidence = self.output_snapshot(), self.transaction_snapshot()
        with self.assertRaisesRegex(ValueError, 'Symlink loop'):
            module.recover_publication(self.root / 'out', [supplied])
        self.assertEqual(self.output_snapshot(), live)
        self.assertEqual(self.transaction_snapshot(), evidence)
        module.recover_publication(self.root / 'out')

    def test_repeated_directory_link_with_parent_path_is_not_a_loop(self):
        real = self.root / 'real'
        real.mkdir()
        source = real / 'input.png'
        Image.fromarray(self.rgb).save(source)
        alias = self.root / 'alias'
        alias.symlink_to('real', target_is_directory=True)
        supplied = alias / '..' / 'alias/input.png'
        original = source.read_bytes()
        report = self.slice_existing(supplied)
        self.assertEqual(report['sheet'], str(source.resolve()))
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue(alias.is_symlink())

    def require_case_alias(self, path, alias):
        if not alias.exists() or not path.samefile(alias):
            self.skipTest('This filesystem treats these case variants as distinct paths.')

    def test_case_alias_destination_recovers_pending_publication(self):
        self.run_cli()
        out, alias = self.root / 'out', self.root / 'OUT'
        self.require_case_alias(out, alias)
        original = self.output_snapshot()
        self.rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(self.rgb).save(self.root / 'sheet.png')
        for point in ('backup_qa', 'install_01.png', 'install_qa'):
            with self.subTest(point=point):
                self.crash_publication(point)
                module.recover_publication(alias)
                self.assertEqual(self.output_snapshot(), original)
                self.assertFalse(list(self.root.glob('.[oO][uU][tT]-*')))

    def test_case_alias_destination_preserves_committed_batch_during_cleanup(self):
        self.run_cli()
        out, alias = self.root / 'out', self.root / 'OUT'
        self.require_case_alias(out, alias)
        original = self.output_snapshot()
        self.rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(self.rgb).save(self.root / 'sheet.png')
        for point in ('journal_committed', 'cleanup_backup', 'cleanup_backup_qa'):
            with self.subTest(point=point):
                self.crash_publication(point)
                committed = self.output_snapshot()
                self.assertNotEqual(committed['01.png'], original['01.png'])
                module.recover_publication(alias)
                self.assertEqual(self.output_snapshot(), committed)
                self.assertFalse(list(self.root.glob('.[oO][uU][tT]-*')))

    def test_case_alias_recovery_can_be_interrupted_and_resumed(self):
        self.run_cli()
        out, alias = self.root / 'out', self.root / 'OUT'
        self.require_case_alias(out, alias)
        original = self.output_snapshot()
        self.rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(self.rgb).save(self.root / 'sheet.png')
        for point in ('discard_qa', 'restore_01.png', 'journal_rolled_back'):
            with self.subTest(point=point):
                self.crash_publication('install_qa')
                self.crash_publication(point, recover_only=True, output=alias)
                module.recover_publication(out)
                self.assertEqual(self.output_snapshot(), original)
                self.assertFalse(list(self.root.glob('.[oO][uU][tT]-*')))

    def test_case_alias_cli_rerun_recovers_then_publishes_new_batch(self):
        self.run_cli()
        out, alias = self.root / 'out', self.root / 'OUT'
        self.require_case_alias(out, alias)
        original = self.output_snapshot()
        self.rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(self.rgb).save(self.root / 'sheet.png')
        self.crash_publication('install_01.png')
        self.assertFalse((out / 'qa/report.json').exists())
        report = self.slice_existing(self.root / 'sheet.png', output=alias)
        result = (out / '01.png').read_bytes()
        self.assertNotEqual(result, original['01.png'])
        self.assertEqual(report['cells'][0]['output_sha256'], hashlib.sha256(result).hexdigest())
        self.assertFalse(list(self.root.glob('.[oO][uU][tT]-*')))

    def test_parent_case_alias_recovers_same_transaction(self):
        self.run_cli()
        parent_alias = self.root.with_name(self.root.name.upper())
        self.require_case_alias(self.root, parent_alias)
        original = self.output_snapshot()
        self.rgb[24:104,24:104] = [30,190,60]
        Image.fromarray(self.rgb).save(self.root / 'sheet.png')
        self.crash_publication('install_01.png')
        module.recover_publication(parent_alias / 'out')
        self.assertEqual(self.output_snapshot(), original)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_legacy_transaction_recovers_through_case_alias(self):
        self.run_cli()
        out, alias = self.root / 'out', self.root / 'OUT'
        self.require_case_alias(out, alias)
        for point in ('install_qa', 'journal_committed'):
            with self.subTest(point=point):
                original = self.output_snapshot()
                self.rgb[24:104,24:104] = [30,190,60]
                Image.fromarray(self.rgb).save(self.root / 'sheet.png')
                self.crash_publication(point)
                expected = self.output_snapshot() if point == 'journal_committed' else original
                journal = module.transaction_path(out)
                record = json.loads(journal.read_text())
                record['version'] = 1
                del record['cleanup']
                journal.write_text(json.dumps(record))
                module.recover_publication(alias)
                self.assertEqual(self.output_snapshot(), expected)
                self.assertFalse(list(self.root.glob('.[oO][uU][tT]-*')))

    def test_transaction_destination_requires_same_output_parent_and_journal(self):
        self.run_cli()
        self.crash_publication('install_01.png')
        out = (self.root / 'out').resolve()
        journal = module.transaction_path(out)
        record = json.loads(journal.read_text())
        original = self.output_snapshot()
        other = self.root / 'other'
        other.mkdir()
        different_parent = other / 'out'
        different_parent.symlink_to(out, target_is_directory=True)
        same_parent = self.root / 'linked-out'
        same_parent.symlink_to(out, target_is_directory=True)
        # A copied journal is not the same transaction even when its destination
        # is another link to the same directory.
        copied_journal = module.transaction_path(same_parent)
        copied_journal.write_bytes(journal.read_bytes())
        for target in (None, 123, 'out', str(other), str(different_parent), str(same_parent)):
            with self.subTest(destination=target):
                record['destination'] = target
                journal.write_text(json.dumps(record))
                evidence = self.transaction_snapshot()
                with self.assertRaisesRegex(ValueError, 'invalid transaction (header|destination)'):
                    module.recover_publication(out)
                self.assertEqual(self.output_snapshot(), original)
                self.assertEqual(self.transaction_snapshot(), evidence)
                self.assertTrue(different_parent.is_symlink())
                self.assertTrue(same_parent.is_symlink())
                self.assertTrue(copied_journal.is_file())

    def test_case_alias_source_cannot_be_replaced_by_preview(self):
        self.run_cli()
        source = self.root / 'out/qa/source.png'
        Image.fromarray(self.rgb).save(source)
        supplied = self.root / 'out/QA/SOURCE.png'
        self.require_case_alias(source, supplied)
        original = self.output_snapshot()
        source_bytes = supplied.read_bytes()
        with self.assertRaisesRegex(FileExistsError, 'input'):
            self.slice_existing(supplied)
        self.assertEqual(supplied.read_bytes(), source_bytes)
        self.assertEqual(self.output_snapshot(), original)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_case_alias_multihop_input_links_are_preserved(self):
        self.run_cli()
        intermediate = self.root / 'out/qa/source.png'
        intermediate.unlink()
        intermediate.symlink_to('../../sheet.png')
        alias = self.root / 'out/QA/SOURCE.png'
        self.require_case_alias(intermediate, alias)
        supplied = self.root / 'input.png'
        supplied.symlink_to('out/QA/SOURCE.png')
        original = self.output_snapshot()
        source_bytes = supplied.read_bytes()
        with self.assertRaisesRegex(FileExistsError, 'input'):
            self.slice_existing(supplied)
        self.assertTrue(intermediate.is_symlink())
        self.assertTrue(supplied.is_symlink())
        self.assertEqual(supplied.read_bytes(), source_bytes)
        self.assertEqual(self.output_snapshot(), original)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_case_alias_review_and_repair_inputs_cannot_be_replaced(self):
        first = self.run_cli()
        review = self.review_file(first)
        repairs = self.root / 'repairs.json'
        repairs.write_text(json.dumps({'cells': {}}))
        source = self.root / 'out/qa/details/01.json'
        supplied = self.root / 'out/QA/DETAILS/01.json'
        self.require_case_alias(source, supplied)
        for option, document in (('--review-file', review), ('--repairs', repairs)):
            with self.subTest(option=option):
                source.write_bytes(document.read_bytes())
                original = self.output_snapshot()
                with self.assertRaisesRegex(FileExistsError, 'input'):
                    self.slice_existing(self.root / 'sheet.png', option, str(supplied))
                self.assertEqual(supplied.read_bytes(), document.read_bytes())
                self.assertEqual(self.output_snapshot(), original)
                self.assertFalse(list(self.root.glob('.out-*')))

    def test_case_alias_unchanged_qa_input_can_still_be_used(self):
        self.run_cli()
        source = self.root / 'out/qa/personal-source.png'
        Image.fromarray(self.rgb).save(source)
        supplied = self.root / 'out/QA/PERSONAL-SOURCE.png'
        self.require_case_alias(source, supplied)
        original = supplied.read_bytes()
        report = self.slice_existing(supplied)
        self.assertEqual(report['exported'], 1)
        self.assertEqual(supplied.read_bytes(), original)
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_case_alias_transaction_inputs_block_cleanup(self):
        for point, directory, leaf in (
            ('install_01.png', 'backup', '01.png'),
            ('journal_committed', 'backup', '01.png'),
            ('install_01.png', 'stage', 'qa/source.png'),
        ):
            with self.subTest(point=point, directory=directory):
                self.run_cli()
                self.require_case_alias(self.root / 'out/qa', self.root / 'out/QA')
                self.crash_publication(point)
                root = next(self.root.glob(f'.out-{directory}-*'))
                source = root / leaf
                supplied = root.with_name(root.name.upper()) / leaf.upper()
                self.require_case_alias(source, supplied)
                source_bytes = supplied.read_bytes()
                live, evidence = self.output_snapshot(), self.transaction_snapshot()
                try:
                    with self.assertRaisesRegex(FileExistsError, 'input'):
                        self.slice_existing(supplied)
                    self.assertEqual(supplied.read_bytes(), source_bytes)
                    self.assertEqual(self.output_snapshot(), live)
                    self.assertEqual(self.transaction_snapshot(), evidence)
                finally:
                    module.recover_publication(self.root / 'out')
                self.assertFalse(list(self.root.glob('.out-*')))

    def test_case_alias_live_qa_input_blocks_rollback_overwrite(self):
        self.run_cli()
        self.require_case_alias(self.root / 'out/qa', self.root / 'out/QA')
        old_source = (self.root / 'out/qa/source.png').read_bytes()
        self.rgb[24:104, 24:104] = [30,190,60]
        Image.fromarray(self.rgb).save(self.root / 'sheet.png')
        self.crash_publication('install_qa')
        supplied = self.root / 'out/QA/SOURCE.png'
        self.assertNotEqual(supplied.read_bytes(), old_source)
        live, evidence = self.output_snapshot(), self.transaction_snapshot()
        with self.assertRaisesRegex(FileExistsError, 'input'):
            module.recover_publication(self.root / 'out', [supplied])
        self.assertEqual(self.output_snapshot(), live)
        self.assertEqual(self.transaction_snapshot(), evidence)
        module.recover_publication(self.root / 'out')
        self.assertEqual((self.root / 'out/qa/source.png').read_bytes(), old_source)

    def test_distinct_case_directory_is_not_treated_as_qa(self):
        self.run_cli()
        separate = self.root / 'out/QA'
        if separate.exists():
            self.skipTest('Distinct case directories require a case-sensitive filesystem.')
        separate.mkdir()
        source = separate / 'source.png'
        Image.fromarray(self.rgb).save(source)
        original = source.read_bytes()
        report = self.slice_existing(source)
        self.assertEqual(report['exported'], 1)
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_each_input_read_stays_locked_after_recovery_returns(self):
        # The other concurrency test covers a reader arriving during publish.
        # This probes the previously exposed gap *after* recovery returned.
        first = self.run_cli()
        review = self.review_file(first)
        repairs = self.root / 'repairs.json'
        repairs.write_text(json.dumps({'cells': {}}))
        probe = r"""
import os, sys
from pathlib import Path
with Path(sys.argv[1]).open('a+b') as handle:
    if os.name == 'nt':
        import msvcrt
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            sys.exit(73)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(73)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
sys.exit(0)
"""
        checked = []
        def assert_locked(label, function):
            def read(*args, **kwargs):
                result = subprocess.run([sys.executable, '-c', probe, str(self.root / '.out.lock')],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 73, f'{label} was read without the output lock: {result.stderr}')
                checked.append(label)
                return function(*args, **kwargs)
            return read
        with patch.object(module.Image, 'open', assert_locked('sheet', module.Image.open)), \
             patch.object(module, 'load_repairs', assert_locked('repairs', module.load_repairs)), \
             patch.object(module, 'load_review', assert_locked('review', module.load_review)):
            self.slice_existing(self.root / 'sheet.png', '--repairs', str(repairs), '--review-file', str(review))
        self.assertEqual(checked, ['sheet', 'repairs', 'review'])

    def test_concurrent_publishers_hold_lock_through_qa_input_reads(self):
        self.run_cli()
        out = (self.root / 'out').resolve()
        source = out / 'qa/personal-source.png'
        Image.fromarray(self.rgb).save(source)
        ready, release = self.root / 'ready', self.root / 'release'
        attempted, opened = self.root / 'attempted', self.root / 'opened'
        child = r"""
import contextlib, importlib.util, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location('slice_icons', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
role, source, output, ready, release, attempted, opened = sys.argv[2:]
output = Path(output)
if role == 'publisher':
    original = Path.replace
    def replace(path, target):
        result = original(path, target)
        if path == output / 'qa' and '-backup-' in Path(target).parent.name:
            Path(ready).touch()
            deadline = time.monotonic() + 15
            while not Path(release).exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('publication barrier timed out')
                time.sleep(0.01)
        return result
    Path.replace = replace
else:
    lock = m.publication_lock
    @contextlib.contextmanager
    def tracked_lock(destination):
        Path(attempted).touch()
        with lock(destination):
            yield
    m.publication_lock = tracked_lock
    open_image = m.Image.open
    def tracked_open(*args, **kwargs):
        Path(opened).touch()
        return open_image(*args, **kwargs)
    m.Image.open = tracked_open
sys.argv = ['slice_icons', source, output.as_posix(), '--grid', '1', '--size', '128']
m.main()
"""
        processes = []
        def start(role, sheet):
            process = subprocess.Popen(
                [sys.executable, '-c', child, module.__file__, role, str(sheet), str(out),
                 str(ready), str(release), str(attempted), str(opened)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            processes.append(process)
            return process
        def wait_for(marker, process):
            deadline = time.monotonic() + 10
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists(), f'Child never reached {marker.name}')
        try:
            publisher = start('publisher', self.root / 'sheet.png')
            wait_for(ready, publisher)
            self.assertFalse(source.exists())
            reader = start('reader', source)
            wait_for(attempted, reader)
            with self.assertRaises(subprocess.TimeoutExpired):
                reader.wait(timeout=0.2)
            self.assertFalse(opened.exists(), 'Input read escaped the publication lock')
            release.touch()
            for process in processes:
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, stdout + stderr)
            self.assertTrue(source.exists())
            self.assertTrue(opened.exists())
            report = json.loads((out / 'qa/report.json').read_text())
            self.assertEqual(report['sheet'], str(source))
            self.assertFalse(list(self.root.glob('.out-*')))
        finally:
            release.touch()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)

    def test_invalid_transaction_cannot_escape_its_output_directory(self):
        self.accepted_grid()
        self.crash_publication('install_01.png', '--grid', '2')
        journal = module.transaction_path(self.root / 'out')
        record = json.loads(journal.read_text())
        partial = self.output_snapshot()
        record['backup'] = '../outside'
        journal.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'invalid backup'):
            module.recover_publication(self.root / 'out')
        self.assertEqual(self.output_snapshot(), partial)

    def test_numbered_source_and_unmanaged_png_survive_publication(self):
        out = self.root / 'out'
        out.mkdir()
        source = out / '20261105.png'
        Image.fromarray(self.rgb).save(source)
        Image.new('RGB', (8,8), 'red').save(out / '99.png')
        original = self.output_snapshot()
        report = self.slice_existing(source)
        self.assertTrue((out / '01.png').exists())
        self.assertTrue(Path(report['sheet']).exists())
        for name, data in original.items():
            self.assertEqual((out / name).read_bytes(), data)

    def test_smaller_batch_removes_only_manifest_owned_pngs(self):
        self.accepted_grid()
        out = self.root / 'out'
        Image.new('RGB', (8,8), 'red').save(out / '99.png')
        extra = (out / '99.png').read_bytes()
        self.run_cli()
        self.assertEqual(sorted(p.name for p in out.glob('*.png')), ['01.png', '99.png'])
        self.assertEqual((out / '99.png').read_bytes(), extra)

    def test_former_output_used_as_source_survives_stale_cleanup(self):
        self.accepted_grid()
        source = self.root / 'out/02.png'
        original = source.read_bytes()
        # Force colour mode because the cropped icon has no uniform backdrop.
        repairs = self.root / 'repairs.json'
        repairs.write_text(json.dumps({'cells': {'1': {'mode': 'floodfill'}}}))
        self.slice_existing(source, '--repairs', str(repairs))
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse((self.root / 'out/03.png').exists())
        self.assertFalse((self.root / 'out/04.png').exists())

    def test_unmanaged_output_name_collision_preserves_all_files(self):
        out = self.root / 'out'
        out.mkdir()
        Image.new('RGB', (8,8), 'red').save(out / '01.png')
        original = self.output_snapshot()
        with self.assertRaisesRegex(FileExistsError, '01.png'):
            self.run_cli()
        self.assertEqual(self.output_snapshot(), original)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_managed_output_cannot_overwrite_its_input_source(self):
        self.run_cli()
        source = self.root / 'out/01.png'
        original = self.output_snapshot()
        repairs = self.root / 'repairs.json'
        repairs.write_text(json.dumps({'cells': {'1': {'mode': 'floodfill'}}}))
        with self.assertRaisesRegex(FileExistsError, 'input'):
            self.slice_existing(source, '--repairs', str(repairs))
        self.assertEqual(self.output_snapshot(), original)
        self.assertFalse(list(self.root.glob('.out-*')))

    def test_modified_stale_output_is_preserved(self):
        self.accepted_grid()
        edited = self.root / 'out/02.png'
        Image.new('RGB', (8,8), 'red').save(edited)
        original = edited.read_bytes()
        self.run_cli()
        self.assertEqual(edited.read_bytes(), original)
        self.assertFalse((self.root / 'out/03.png').exists())

    def test_legacy_report_without_hashes_still_identifies_old_exports(self):
        self.accepted_grid()
        report_path = self.root / 'out/qa/report.json'
        report = json.loads(report_path.read_text())
        for cell in report['cells']:
            del cell['output_sha256']
        report_path.write_text(json.dumps(report))
        self.run_cli()
        self.assertEqual([p.name for p in (self.root / 'out').glob('*.png')], ['01.png'])

    def test_invalid_manifest_filename_cannot_remove_user_files(self):
        self.run_cli()
        report_path = self.root / 'out/qa/report.json'
        report = json.loads(report_path.read_text())
        report['cells'][0]['file'] = '../sheet.png'
        report_path.write_text(json.dumps(report))
        original = self.output_snapshot()
        source = (self.root / 'sheet.png').read_bytes()
        with self.assertRaisesRegex(ValueError, 'report'):
            self.run_cli()
        self.assertEqual(self.output_snapshot(), original)
        self.assertEqual((self.root / 'sheet.png').read_bytes(), source)

    def test_clean_output_still_needs_visual_acceptance(self):
        self.rgb[44:64,44:64] = [240,210,160]
        first = self.run_cli()
        self.assertFalse(first['cells'][0]['findings'])
        self.assertEqual(first['cells'][0]['review']['status'], 'pending')
        self.assertEqual(first['review_cells'], [1])
        path = self.review_file(first, decisions={})
        accepted = self.run_cli('--review-file', str(path))
        self.assertEqual(accepted['status'], 'accepted')
        # Omitting the record on a later run must never silently approve it.
        self.assertEqual(self.run_cli()['status'], 'needs_review')

    def test_invalid_and_duplicate_review_documents_preserve_existing_outputs(self):
        first = self.run_cli()
        path = self.review_file(first)
        valid = json.loads(path.read_text())
        bad = []
        for field, value in [('status', 'ok'), ('note',' '), ('reviewer',False), ('fingerprint','bad'),
                             ('decisions',{'unknown':'valid_hole'}), ('decisions',{'transparent_holes':'needs_fix'})]:
            doc = json.loads(json.dumps(valid))
            doc['cells']['1'][field] = value
            bad.append(json.dumps(doc))
        bad += ['{"cells":{},"cells":{}}', '{"cells":{"1":{},"1":{}}}', 'null']
        files = {str(p): p.read_bytes() for p in (self.root/'out').rglob('*') if p.is_file()}
        for text in bad:
            path.write_text(text)
            with self.subTest(text=text), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_cli('--review-file', str(path))
            self.assertEqual(files, {str(p):p.read_bytes() for p in (self.root/'out').rglob('*') if p.is_file()})

    def test_noisy_sixteen_cell_report_is_bounded_and_details_are_complete(self):
        sheet = np.full((512,512,3), self.bg, dtype=np.uint8)
        for row in range(4):
            for col in range(4):
                x,y=col*128,row*128
                sheet[y+20:y+108,x+20:x+108] = [240,210,160]
                for iy in range(8):
                    for ix in range(8):
                        sx,sy=x+26+ix*9,y+26+iy*9
                        sheet[sy:sy+3,sx:sx+3] = self.bg
                        sheet[sy+4:sy+7,sx:sx+3] = self.bg+8
        report = self.run_cli('--grid','4', rgb=sheet)
        text = (self.root/'out/qa/report.json').read_text()
        summary = (self.root/'out/qa/summary.md').read_text()
        self.assertLess(len(text.splitlines()), 80)
        self.assertLess(len(summary.splitlines()), 40)
        self.assertLess(len(text.encode()), 80000)
        for cell in report['cells']:
            details = json.loads((self.root/'out'/cell['details']).read_text())
            hole = next(f for f in details['findings'] if f['code']=='transparent_holes')
            self.assertEqual(hole['region_count'],64)
            self.assertEqual(len(hole['regions']),64)
            self.assertEqual(hole['pixels'],sum(region['area'] for region in hole['regions']))
            for finding in cell['findings']:
                self.assertLessEqual(len(finding['regions']),3)
        self.run_cli()
        self.assertEqual([p.name for p in (self.root/'out/qa/details').glob('*.json')], ['01.json'])

    def test_faint_edge_hints_do_not_hide_strong_thin_detail_loss(self):
        rgb = self.rgb.copy()
        rgb[44:64,44:64] = [240,210,160]
        rgb[24:104,23] = self.bg+8
        rgb[50:53,104:115] = [30,190,60]
        alpha = module.floodfill_alpha(rgb, bg=self.bg, tolerance=3)
        alpha[24:104,23] = 0
        alpha[50:53,104:115] = 0
        report = module.inspect_cell(rgb,rgb,alpha,alpha,(0,0,128,128),module.estimate_background(rgb),24)
        loss = next(f for f in report['findings'] if f['code']=='possible_cutout_loss')
        edge = next(f for f in report['findings'] if f['code']=='edge_ambiguity')
        self.assertEqual(loss['pixels'],33)
        self.assertEqual(edge['pixels'],80)
        self.assertEqual(edge['severity'],'info')


if __name__ == '__main__':
    unittest.main()
