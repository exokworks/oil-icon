#!/usr/bin/env python3
"""Verify all installed image dependencies, the U2-Net model and CPU inference."""

import io
import os
import platform
import sys
from importlib.metadata import version


def main():
    print(f'Checking Python {platform.python_version()} ({platform.machine()})', flush=True)

    import numpy as np
    from PIL import Image, ImageDraw
    from scipy import ndimage

    for package in ['Pillow', 'numpy', 'scipy']:
        print(f'  {package}=={version(package)}', flush=True)
    image = Image.new('RGB', (64, 64), (128, 128, 128))
    ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill=(210, 60, 40))
    mask = np.any(np.array(image) != 128, axis=2)
    _, components = ndimage.label(mask)
    if components != 1:
        raise RuntimeError('NumPy/SciPy image processing check failed')
    encoded = io.BytesIO()
    image.save(encoded, format='PNG')
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        decoded.load()
        if decoded.size != image.size:
            raise RuntimeError('Pillow PNG round-trip failed')
    print('  Base image processing passed', flush=True)

    # Keep the installation smoke check small; callers may override this.
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    from slice_icons import create_rembg_session
    from rembg import remove

    for package in ['rembg', 'onnxruntime', 'numba', 'llvmlite']:
        print(f'  {package}=={version(package)}', flush=True)
    print('  Preparing U2-Net; first use downloads ~176 MB from GitHub', flush=True)
    session = create_rembg_session()
    if session.inner_session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('The session did not select the CPU provider')
    result = remove(image, session=session)
    if result.mode != 'RGBA' or result.size != image.size:
        raise RuntimeError('rembg did not return an RGBA image of the expected size')
    print(f'  Model cache: {os.environ.get("U2NET_HOME") or os.environ.get("REMBG_HOME")}', flush=True)
    print('  U2-Net CPU inference passed', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Environment verification failed: {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
