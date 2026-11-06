"""Exercise the installer in an isolated Skill folder without network or models."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='oil-icon setup ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.skill = self.root / 'installed skill'
        scripts = self.skill / 'scripts'
        scripts.mkdir(parents=True)
        self.setup = scripts / 'setup.sh'
        shutil.copy2(ROOT / 'scripts/setup.sh', self.setup)
        shutil.copy2(ROOT / 'requirements.txt', self.skill / 'requirements.txt')
        # Keep Python/venv real; replace pip and inference at the network boundary.
        stubs = self.root / 'stubs'
        stubs.mkdir()
        (stubs / 'pip.py').write_text('''import json, os, subprocess, sys
with open(os.environ['SETUP_PIP_LOG'], 'a') as log:
    log.write(json.dumps({'args': sys.argv[1:], 'prefix': sys.prefix}) + '\\n')
if sys.argv[1:] == ['config', 'list']:
    sys.exit(subprocess.call([sys.executable, '-I', '-m', 'pip', 'config', 'list']))
if 'install' in sys.argv and os.environ.get('SETUP_FAIL_INSTALL'):
    print('simulated wheel resolution failure', file=sys.stderr)
    sys.exit(9)
''')
        (scripts / 'check_env.py').write_text('''import os
with open(os.environ['SETUP_CHECK_LOG'], 'a') as log:
    log.write('checked\\n')
''')
        self.pip_log = self.root / 'pip.jsonl'
        self.check_log = self.root / 'checks.txt'
        self.env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PYTHONPATH=str(stubs),
                        SETUP_PIP_LOG=str(self.pip_log), SETUP_CHECK_LOG=str(self.check_log))
        for variable in ['PIP_TARGET', 'PIP_PREFIX', 'PIP_ROOT', 'PIP_USER']:
            self.env.pop(variable, None)
        self.caller = self.root / 'different working directory'
        self.caller.mkdir()

    def run_setup(self, *args):
        return subprocess.run([str(self.setup), *args], cwd=self.caller, env=self.env,
                              capture_output=True, text=True, timeout=60)

    def require_supported_python(self):
        if not (3, 12) <= sys.version_info[:2] < (3, 14):
            self.skipTest('installer integration requires Python 3.12 or 3.13')

    def test_full_install_reuses_venv_and_ignores_broken_pip_entrypoint(self):
        self.require_supported_python()
        first = self.run_setup('--python', sys.executable)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        venv = self.skill / '.venv'
        marker = venv / 'preserve-me.txt'
        marker.write_text('keep existing environment')
        (venv / 'bin/pip').write_text('#!/nonexistent/old/location/python\n')
        second = self.run_setup()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        # The ready message must leave edge-based selection to the agent;
        # quoting must also keep paths with spaces runnable when copied.
        for result in (first, second):
            commands = [line.removeprefix('Slice with: ') for line in result.stdout.splitlines()
                        if line.startswith('Slice with: ')]
            self.assertEqual(len(commands), 1)
            self.assertEqual(shlex.split(commands[0]), [
                str(venv.resolve() / 'bin/python'),
                str(self.skill.resolve() / 'scripts/slice_icons.py'), 'sheet.png', 'out',
            ])
            self.assertIn('Agent selects cutout from actual image edges', result.stdout)
        self.assertEqual(marker.read_text(), 'keep existing environment')
        self.assertFalse((self.caller / '.venv').exists())
        self.assertEqual(self.check_log.read_text().splitlines(), ['checked', 'checked'])
        calls = [json.loads(line) for line in self.pip_log.read_text().splitlines()]
        installs = [call for call in calls if call['args'][0] == 'install']
        self.assertEqual(len(installs), 2)
        for call in installs:
            self.assertEqual(Path(call['prefix']).resolve(), venv.resolve())
            self.assertIn('--only-binary=:all:', call['args'])
            self.assertIn('--no-user', call['args'])
            requirements = call['args'][call['args'].index('-r') + 1]
            self.assertEqual(Path(requirements).resolve(), (self.skill / 'requirements.txt').resolve())
        self.assertEqual(sum(call['args'] == ['check'] for call in calls), 2)

    def test_pip_location_environment_is_rejected_before_installing(self):
        self.require_supported_python()
        for variable in ['PIP_TARGET', 'PIP_PREFIX', 'PIP_ROOT']:
            with self.subTest(variable=variable):
                outside = self.root / 'outside installation'
                self.env[variable] = str(outside)
                result = self.run_setup('--python', sys.executable)
                self.env.pop(variable)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(variable[4:].lower(), result.stderr)
                self.assertNotIn('environment ready', result.stdout)
                self.assertFalse(outside.exists())
        calls = [json.loads(line) for line in self.pip_log.read_text().splitlines()]
        self.assertFalse(any(call['args'][0] == 'install' for call in calls))
        self.assertFalse(self.check_log.exists())

    def test_pip_location_config_is_rejected_without_disclosing_values(self):
        self.require_supported_python()
        config = self.root / 'custom pip.conf'
        self.env['PIP_CONFIG_FILE'] = str(config)
        for section, option in [('global', 'target'), ('install', 'prefix'), ('install', 'root')]:
            with self.subTest(section=section, option=option):
                config.write_text(f'[{section}]\n{option} = /private-location-marker\n')
                result = self.run_setup('--python', sys.executable)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f'{section}.{option}', result.stderr)
                self.assertNotIn('/private-location-marker', result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.pip_log.read_text().splitlines()]
        self.assertFalse(any(call['args'][0] == 'install' for call in calls))
        self.assertFalse(self.check_log.exists())

    def test_unsupported_interpreter_fails_before_creating_environment(self):
        unsupported = self.root / 'unsupported python'
        unsupported.write_text('#!/bin/sh\nexit 1\n')
        unsupported.chmod(0o755)
        result = self.run_setup('--python', str(unsupported))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python 3.12 or 3.13', result.stderr)
        self.assertFalse((self.skill / '.venv').exists())
        self.assertFalse(self.pip_log.exists())

    def test_invalid_existing_directory_is_preserved(self):
        venv = self.skill / '.venv'
        venv.mkdir()
        marker = venv / 'preserve-me.txt'
        marker.write_text('do not delete')
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Move it aside', result.stderr)
        self.assertEqual(marker.read_text(), 'do not delete')
        self.assertFalse(self.pip_log.exists())

    def test_install_failure_does_not_report_ready_or_run_inference(self):
        self.require_supported_python()
        self.env['SETUP_FAIL_INSTALL'] = '1'
        result = self.run_setup('--python', sys.executable)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('installing all image dependencies', result.stderr)
        self.assertNotIn('environment ready', result.stdout)
        self.assertFalse(self.check_log.exists())
        calls = [json.loads(line) for line in self.pip_log.read_text().splitlines()]
        self.assertFalse(any(call['args'] == ['check'] for call in calls))


if __name__ == '__main__':
    unittest.main()
