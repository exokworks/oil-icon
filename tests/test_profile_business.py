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
from contextlib import redirect_stdout
spec=importlib.util.spec_from_file_location('icons',Path(__file__).parents[1]/'scripts/gen_image.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class CredentialTests(unittest.TestCase):
    def test_actual_client_consumes_profile_key_without_printing_it(self):
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
             {'model': 'gpt-image-1', 'prompt': 'test', 'size': '1024x1024', 'n': 1}),
            ('/proxy/v1/images/generations', 'Bearer TEST_ONLY_ICON',
             'Custom-Image-Client/1.0',
             {'model': 'gpt-image-1', 'prompt': 'test', 'size': '1024x1024', 'n': 1}),
        ])
