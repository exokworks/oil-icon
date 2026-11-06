import importlib.util
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys
import tempfile
import unittest
from threading import Thread
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
spec=importlib.util.spec_from_file_location('icons',Path(__file__).parents[1]/'scripts/gen_image.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class GenImageTests(unittest.TestCase):
    def test_long_utf8_file_prompt_reaches_both_requests_unchanged(self):
        prompt = '  一套图标，保留中文和换行。\n' + ('细节：耳机、叶片、星星；不要文字。\n' * 400) + '\n'
        requests = []

        def response(request, **kwargs):
            requests.append(json.loads(request.data))
            if len(requests) == 1:
                raise module.urllib.error.HTTPError(
                    'https://example.invalid', 403, 'Forbidden', {}, io.BytesIO())
            return io.BytesIO(json.dumps({'data': [{'b64_json': 'aWNvbg=='}]}).encode())

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'long prompt.txt'
            source.write_text(prompt, encoding='utf-8-sig')
            output = Path(directory) / 'out.png'
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON'}), \
                    patch.object(sys, 'argv', ['gen_image', '--prompt-file', str(source), '--out', str(output)]), \
                    patch.object(module.urllib.request, 'urlopen', side_effect=response), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                module.main()
            self.assertEqual(output.read_bytes(), b'icon')
            self.assertEqual([request['prompt'] for request in requests], [prompt, prompt])
            self.assertEqual([request['model'] for request in requests], [module.DEFAULT_MODEL, module.FALLBACK_MODEL])
            self.assertEqual(source.read_text(encoding='utf-8-sig'), prompt)

    def test_prompt_output_aliases_fail_before_request_and_preserve_prompt(self):
        for alias in ('same_path', 'symlink', 'hardlink', 'directory_symlink'):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'prompt.txt'
                original = '完整提示词，必须保留以便复现。\n'.encode('utf-8')
                source.write_bytes(original)
                output = root / 'out.png'
                if alias == 'same_path':
                    output = source
                elif alias == 'symlink':
                    output.symlink_to(source)
                elif alias == 'hardlink':
                    output.hardlink_to(source)
                else:
                    link = root / 'linked-directory'
                    link.symlink_to(root, target_is_directory=True)
                    output = link / source.name
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON'}), \
                        patch.object(sys, 'argv', ['gen_image', '--prompt-file', str(source), '--out', str(output)]), \
                        patch.object(module, 'generate_image', return_value=b'icon') as request, \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as error, \
                        self.assertRaises(SystemExit) as failure:
                    module.main()
                self.assertEqual(failure.exception.code, 2)
                self.assertIn('--out must not refer to the --prompt-file input', error.getvalue())
                request.assert_not_called()
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(output.read_bytes(), original)

    def test_prompt_file_allows_distinct_output_with_identical_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'prompt.txt', root / 'out.png'
            prompt = '保留提示词，替换独立的输出文件。\n'
            source.write_text(prompt, encoding='utf-8')
            output.write_bytes(source.read_bytes())
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON'}), \
                    patch.object(sys, 'argv', ['gen_image', '--prompt-file', str(source), '--out', str(output)]), \
                    patch.object(module, 'generate_image', return_value=b'icon') as request, \
                    redirect_stdout(io.StringIO()):
                module.main()
            request.assert_called_once()
            self.assertEqual(request.call_args.args[1], prompt)
            self.assertEqual(source.read_text(encoding='utf-8'), prompt)
            self.assertEqual(output.read_bytes(), b'icon')

    def test_invalid_prompt_inputs_fail_before_request_and_preserve_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'prompt.txt'
            source.write_text('提示词', encoding='utf-8')
            empty = root / 'empty.txt'
            empty.write_text(' \n\t', encoding='utf-8')
            invalid = root / 'invalid.txt'
            invalid.write_bytes(b'\xff\xfe')
            output = root / 'out.png'
            output.write_bytes(b'previous output')
            cases = [
                [], ['--prompt', ''], ['--prompt', ' \n\t'],
                ['--prompt-file', str(empty)], ['--prompt-file', str(invalid)],
                ['--prompt-file', str(root / 'missing.txt')], ['--prompt-file', str(root)],
                ['--prompt', 'inline', '--prompt-file', str(source)],
            ]
            for args in cases:
                with self.subTest(args=args), \
                        patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON'}), \
                        patch.object(sys, 'argv', ['gen_image', *args, '--out', str(output)]), \
                        patch.object(module.urllib.request, 'urlopen') as request, \
                        redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                    module.main()
                self.assertEqual(failure.exception.code, 2)
                request.assert_not_called()
                self.assertEqual(output.read_bytes(), b'previous output')

    def test_environment_key_authorizes_image_request_without_printing_it(self):
        def response(request,**kwargs):
            self.assertEqual(request.get_header('Authorization'),'Bearer TEST_ONLY_ICON')
            self.assertEqual(request.get_header('User-agent'), 'Mozilla/5.0 (compatible; oil-icon/1.0)')
            self.assertEqual(request.full_url, 'https://api.openai.com/v1/images/generations')
            return io.BytesIO(json.dumps({'data':[{'b64_json':'dGVzdA=='}]}).encode())
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'OPENAI_API_KEY':'TEST_ONLY_ICON','OPENAI_BASE_URL':'','OPENAI_USER_AGENT':''}),patch.object(sys,'argv',['gen_image','--prompt','test','--out',str(Path(directory)/'out.png')]),patch.object(module.urllib.request,'urlopen',side_effect=response),redirect_stdout(io.StringIO()) as out:
            module.main()
            self.assertNotIn('TEST_ONLY_ICON',out.getvalue())
            self.assertEqual((Path(directory)/'out.png').read_bytes(),b'test')

    def test_custom_base_url_reaches_image_endpoint(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append((self.path, self.headers.get('Authorization'), self.headers.get('User-Agent'), json.loads(
                    self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'data': [{'b64_json': 'aWNvbg=='}]}).encode())

            def log_message(self, *_args):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever)
        thread.start()
        original_urlopen = module.urllib.request.build_opener(
            module.urllib.request.ProxyHandler({})).open
        base = f'http://127.0.0.1:{server.server_port}/proxy/v1'

        def local_only(request, **kwargs):
            self.assertTrue(request.full_url.startswith(base), 'request must stay on the test server')
            return original_urlopen(request, **kwargs)

        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'out.png'
                for base_url, user_agent in ((base, ''), (base + '/', 'Custom-Image-Client/1.0')):
                    with patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON', 'OPENAI_BASE_URL': base_url, 'OPENAI_USER_AGENT': user_agent}), \
                            patch.object(sys, 'argv', ['gen_image', '--prompt', 'test', '--out', str(output)]), \
                            patch.object(module.urllib.request, 'urlopen', side_effect=local_only), \
                            redirect_stdout(io.StringIO()):
                        module.main()
                    self.assertEqual(output.read_bytes(), b'icon')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(requests, [
            ('/proxy/v1/images/generations', 'Bearer TEST_ONLY_ICON',
             'Mozilla/5.0 (compatible; oil-icon/1.0)',
             {'model': 'gpt-image-2.6-sunburst', 'prompt': 'test', 'size': '1024x1024', 'n': 1}),
            ('/proxy/v1/images/generations', 'Bearer TEST_ONLY_ICON',
             'Custom-Image-Client/1.0',
             {'model': 'gpt-image-2.6-sunburst', 'prompt': 'test', 'size': '1024x1024', 'n': 1}),
        ])

    def test_2_6_rejection_retries_2_5_sunburst_without_leaking_response(self):
        error = module.urllib.error.HTTPError(
            'https://example.invalid', 400, 'Bad Request', {}, io.BytesIO(b'secret response detail')
        )
        requests = []

        def response(request, **kwargs):
            requests.append(json.loads(request.data))
            if len(requests) == 1:
                raise error
            return io.BytesIO(json.dumps({'data': [{'b64_json': 'dGVzdA=='}]}).encode())

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON'}), \
                patch.object(sys, 'argv', ['gen_image', '--prompt', 'test', '--out', str(Path(directory) / 'out.png')]), \
                patch.object(module.urllib.request, 'urlopen', side_effect=response), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as notice:
            module.main()
            self.assertEqual([request['model'] for request in requests], [
                'gpt-image-2.6-sunburst', 'gpt-image-2.5-sunburst'
            ])
            self.assertEqual((Path(directory) / 'out.png').read_bytes(), b'test')
            self.assertIn('used fallback gpt-image-2.5-sunburst', notice.getvalue())
            self.assertNotIn('secret response detail', notice.getvalue())
            self.assertNotIn('TEST_ONLY_ICON', notice.getvalue())

    def test_flare_rejection_automatically_uses_2_5_sunburst(self):
        error = module.urllib.error.HTTPError('https://example.invalid', 404, 'Not Found', {}, io.BytesIO())
        requests = []

        def response(request, **kwargs):
            requests.append(json.loads(request.data)['model'])
            if len(requests) == 1:
                raise error
            return io.BytesIO(json.dumps({'data': [{'b64_json': 'aWNvbg=='}]}).encode())

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON'}), \
                patch.object(sys, 'argv', ['gen_image', '--prompt', 'test', '--out', str(Path(directory) / 'out.png'), '--model', 'gpt-image-2.6-flare']), \
                patch.object(module.urllib.request, 'urlopen', side_effect=response), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            module.main()
            self.assertEqual(requests, ['gpt-image-2.6-flare', 'gpt-image-2.5-sunburst'])
            self.assertEqual((Path(directory) / 'out.png').read_bytes(), b'icon')

    def test_403_rejection_automatically_uses_2_5_sunburst(self):
        error = module.urllib.error.HTTPError('https://example.invalid', 403, 'Forbidden', {}, io.BytesIO())
        requests = []

        def response(request, **kwargs):
            requests.append(json.loads(request.data)['model'])
            if len(requests) == 1:
                raise error
            return io.BytesIO(json.dumps({'data': [{'b64_json': 'aWNvbg=='}]}).encode())

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON'}), \
                patch.object(sys, 'argv', ['gen_image', '--prompt', 'test', '--out', str(Path(directory) / 'out.png')]), \
                patch.object(module.urllib.request, 'urlopen', side_effect=response), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            module.main()
            self.assertEqual(requests, ['gpt-image-2.6-sunburst', 'gpt-image-2.5-sunburst'])
            self.assertEqual((Path(directory) / 'out.png').read_bytes(), b'icon')

    def test_2_6_failure_with_explicit_base_name_notes_valid_api_ids(self):
        self.assertTrue(module.uses_2_6('gpt-image-2.6'))
        self.assertTrue(module.uses_2_6('gpt-image-2.6-flare'))
        self.assertTrue(module.uses_2_6('gpt-image-2.6-sunburst'))

    def test_non_2_6_failure_does_not_suggest_unrequested_fallback(self):
        self.assertFalse(module.uses_2_6('custom-image-model'))

    def test_timeout_does_not_retry_to_avoid_duplicate_generation(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON'}), \
                patch.object(sys, 'argv', ['gen_image', '--prompt', 'test', '--out', str(Path(directory) / 'out.png')]), \
                patch.object(module.urllib.request, 'urlopen', side_effect=TimeoutError), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            module.main()
        self.assertIn('Automatic fallback to gpt-image-2.5-sunburst was skipped', str(failure.exception))
        self.assertFalse((Path(directory) / 'out.png').exists())

    def test_failed_automatic_fallback_reports_both_statuses(self):
        first_error = module.urllib.error.HTTPError('https://example.invalid', 404, 'Not Found', {}, io.BytesIO())
        second_error = module.urllib.error.HTTPError('https://example.invalid', 422, 'Unprocessable', {}, io.BytesIO())
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_ONLY_ICON'}), \
                patch.object(sys, 'argv', ['gen_image', '--prompt', 'test', '--out', str(Path(directory) / 'out.png')]), \
                patch.object(module.urllib.request, 'urlopen', side_effect=[first_error, second_error]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            module.main()
        self.assertIn('gpt-image-2.6-sunburst (HTTP 404)', str(failure.exception))
        self.assertIn('gpt-image-2.5-sunburst also failed (HTTP 422)', str(failure.exception))
        self.assertFalse((Path(directory) / 'out.png').exists())
