#!/usr/bin/env python3
"""Slice icon sheets, estimate their backdrop, and export PNGs with QA artifacts.

Flat art uses a measured colour key plus a narrow edge matte. Soft art uses
rembg. Components are assigned by cell ownership, including detached accents.
QA findings are review hints, never a claim of semantic correctness.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


CONNECTED = np.ones((3, 3), dtype=bool)


def colour_distance(rgb, bg):
    return np.abs(np.asarray(rgb, dtype=float) - bg).max(axis=2)


def estimate_background(rgb, grid=1):
    """Find a dominant colour across outer edges AND grid gutters, not corners.

    Quantisation finds the dominant cluster; nearby samples recover its actual
    colour and noise. Agreement across bands guards against a foreground tile
    occupying one corner or one gutter. Low confidence triggers rembg in main.
    """
    h, w = rgb.shape[:2]
    width = max(1, int(min(h, w) / grid * 0.015))
    bands = []
    for x in sorted({min(w - 1, i * w // grid) for i in range(grid + 1)}):
        bands.append(rgb[:, max(0, x - width):min(w, x + width + 1)].reshape(-1, 3))
    for y in sorted({min(h - 1, i * h // grid) for i in range(grid + 1)}):
        bands.append(rgb[max(0, y - width):min(h, y + width + 1), :].reshape(-1, 3))
    samples = np.concatenate(bands).astype(float)
    bins, counts = np.unique((samples // 8).astype(int), axis=0, return_counts=True)
    dominant = bins[counts.argmax()]
    seed = np.median(samples[np.all(samples // 8 == dominant, axis=1)], axis=0)
    inliers = np.max(np.abs(samples - seed), axis=1) <= 12
    colour = np.rint(np.median(samples[inliers], axis=0)).astype(int)
    distances = np.max(np.abs(samples - colour), axis=1)
    tolerance = float(np.clip(np.percentile(distances[inliers], 99) + 1, 2, 6))
    support = float(np.mean(distances <= tolerance))
    agreement = float(np.mean([
        np.mean(np.max(np.abs(band.astype(float) - colour), axis=1) <= tolerance) >= 0.6
        for band in bands
    ]))
    return {
        'rgb': colour.tolist(),
        'hex': '#%02x%02x%02x' % tuple(colour),
        'tolerance': tolerance,
        'sample_support': round(support, 4),
        'band_agreement': round(agreement, 4),
        'confident': support >= 0.7 and agreement >= 0.75,
    }


def floodfill_alpha(rgb, thresh=24, bg=None, tolerance=2):
    """Remove measured background, preserving merely similar interior colours.

    Only tightly matching pixels are removed. Border-connected noise can use
    the measured tolerance; enclosed regions require an almost exact match.
    Exact backdrop-coloured subject details cannot be distinguished from holes
    by colour alone, so QA reports every substantial enclosed transparent area.
    `thresh` caps tolerance; it is no longer a global transparent-colour range.
    """
    if bg is None:
        background = estimate_background(rgb)
        bg, tolerance = background['rgb'], background['tolerance']
    tolerance = min(tolerance, thresh)
    diff = colour_distance(rgb, bg)
    matching = diff <= tolerance
    seeds = np.zeros(matching.shape, dtype=bool)
    seeds[[0, -1], :] = matching[[0, -1], :]
    seeds[:, [0, -1]] = matching[:, [0, -1]]
    exterior = ndimage.binary_propagation(seeds, mask=matching)
    exact_holes = diff <= min(1, tolerance)
    return np.where(exterior | exact_holes, 0, 255).astype(np.uint8)


def refine_edges(rgb, alpha, bg, thresh=24, tolerance=2, infer_alpha=True):
    """Unmix only narrow, well-supported foreground/background colour blends.

    Use a nearby opaque interior colour as a local foreground estimate. Reject
    colour mismatches and distant anchors rather than modifying shading or
    isolated thin marks. For rembg, retain its alpha and require agreement with
    that alpha before removing background colour from RGB.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    corrected, matte = rgb.copy(), alpha.copy()
    foreground = alpha > 0
    if not foreground.any() or foreground.all():
        return corrected, matte
    distance = ndimage.distance_transform_edt(foreground)
    anchors = (alpha >= 250) & (distance >= 3) & (colour_distance(rgb, bg) >= thresh)
    if not anchors.any():
        return corrected, matte
    nearest_distance, nearest = ndimage.distance_transform_edt(~anchors, return_indices=True)
    estimate = rgb[tuple(nearest)].astype(float)
    vector = estimate - bg
    observed = rgb.astype(float) - bg
    denominator = np.sum(vector * vector, axis=2)
    ratio = np.sum(observed * vector, axis=2) / np.maximum(denominator, 1)
    error = np.max(np.abs(observed - ratio[:, :, None] * vector), axis=2)
    candidate = foreground & (distance <= 2) & (nearest_distance <= 4)
    candidate &= (ratio > 0.02) & (ratio < 0.98) & (error <= max(2, tolerance))
    if not infer_alpha:
        candidate &= (alpha < 250) & (np.abs(ratio - alpha / 255) <= 0.1)
    else:
        matte[candidate] = np.clip(np.rint(ratio[candidate] * 255), 1, 254).astype(np.uint8)
    if candidate.any():
        a = matte[candidate, None] / 255.0
        unmixed = (rgb[candidate].astype(float) - (1 - a) * np.asarray(bg)) / a
        corrected[candidate] = np.clip(np.rint(unmixed), 0, 255).astype(np.uint8)
    return corrected, matte


def cell_mask(shape, bounds):
    left, top, right, bottom = bounds
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def clean(alpha, bounds=None):
    """Keep all components belonging to the original cell, even tiny accents.

    Discard components wholly in the padding, or mostly outside the cell and
    clipped by the crop edge. Ambiguous overlaps stay for QA. Separate visible
    cores first so faint rembg bridges do not join neighbouring icons; attach
    soft pixels to their nearest core without dropping detached faint accents.
    """
    mask = alpha > 0
    if not mask.any():
        return None
    h, w = mask.shape
    bounds = bounds or (0, 0, w, h)
    owned = cell_mask(mask.shape, bounds)
    support, support_count = ndimage.label(mask, structure=CONNECTED)
    cores, core_count = ndimage.label(alpha > 32, structure=CONNECTED)
    if core_count:
        nearest = ndimage.distance_transform_edt(cores == 0, return_distances=False, return_indices=True)
        same_support = mask & (support[tuple(nearest)] == support)
        labels = np.where(same_support, cores[tuple(nearest)], core_count + support)
        labels[~mask] = 0
        n = core_count + support_count
    else:
        labels, n = support, support_count
    mass = np.bincount(labels.ravel(), weights=alpha.ravel(), minlength=n + 1)
    inside = np.bincount(labels[owned], weights=alpha[owned], minlength=n + 1)
    touches_edge = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    discard = inside == 0
    discard[touches_edge] |= inside[touches_edge] < 0.5 * mass[touches_edge]
    discard[0] = True
    out = alpha.copy()
    out[discard[labels]] = 0
    return out


def regions(mask, minimum=1):
    labels, n = ndimage.label(mask, structure=CONNECTED)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    result = []
    for label, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if sizes[label] >= minimum:
            ys, xs = bounds
            result.append({'area': int(sizes[label]), 'bbox': [xs.start, ys.start, xs.stop, ys.stop]})
    return result


def inspect_cell(rgb, cutout_rgb, before, after, bounds, background, thresh):
    """Compare source, cutout, and cleanup; report ambiguities without erasing."""
    owned = cell_mask(after.shape, bounds)
    findings = []

    def flag(code, message, mask, minimum=3):
        objects = regions(mask, minimum)
        if objects:
            findings.append({
                'code': code, 'message': message,
                'pixels': sum(obj['area'] for obj in objects),
                'region_count': len(objects), 'regions': objects[:50],
            })

    if not after.any():
        findings.append({'code': 'empty', 'message': 'No foreground remains in this cell.'})
    flag('removed_components', 'Cleanup removed content; check against the source cell.', (before > 0) & (after == 0), 1)
    visible = after > 32
    holes = ndimage.binary_fill_holes(visible) & ~visible
    flag('transparent_holes', 'Enclosed transparency may be a valid hole or lost subject detail.', holes & owned)

    if background['confident']:
        bg, tolerance = background['rgb'], background['tolerance']
        similar = colour_distance(cutout_rgb, bg) <= max(thresh, tolerance)
        flag('background_like_foreground', 'Retained pixels resemble the backdrop; inspect for grey residue or intentional grey detail.', similar & visible & owned)
        # Compare locally so a missing accent attached to a retained body is
        # reported even when both form one source-image component.
        source = (colour_distance(rgb, bg) > tolerance + 2) & owned
        missing = source & (before <= 32)
        flag(
            'possible_cutout_loss',
            'Foreground-coloured source pixels are nearly transparent after cutout; inspect for lost details.',
            missing,
        )

    rim = owned & ~ndimage.binary_erosion(owned)
    flag('cell_boundary', 'Foreground reaches a cell boundary; check clipping or neighbour bleed.', visible & rim)
    return {
        'cutout_components': len(regions(before > 0)),
        'retained_components': len(regions(after > 0)),
        'removed_pixels': int(np.count_nonzero((before > 0) & (after == 0))),
        'findings': findings,
    }


def normalise(rgba, size):
    alpha = np.asarray(rgba)[:, :, 3]
    if not alpha.any():
        return Image.new('RGBA', (size, size))
    ys, xs = np.where(alpha > 0)
    obj = rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    side = max(1, int(max(obj.size) * 1.16))
    canvas = Image.new('RGBA', (side, side))
    canvas.paste(obj, ((side - obj.width) // 2, (side - obj.height) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def save_preview(images, path, grid, background, reports):
    tile, heading = 192, 24
    columns = min(grid, len(images))
    preview = Image.new('RGB', (columns * tile, math.ceil(len(images) / columns) * (tile + heading)), 'white')
    draw = ImageDraw.Draw(preview)
    for index, image in enumerate(images):
        x, y = (index % columns) * tile, (index // columns) * (tile + heading)
        mark = ' REVIEW' if reports[index]['findings'] else ''
        draw.text((x + 5, y + 5), f'{index + 1:02d}{mark}', fill='#b22222' if mark else '#222222')
        thumbnail = image.convert('RGBA')
        thumbnail.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        panel = Image.new('RGBA', (tile, tile), background)
        panel.alpha_composite(thumbnail, ((tile - thumbnail.width) // 2, (tile - thumbnail.height) // 2))
        preview.paste(panel.convert('RGB'), (x, y + heading))
    preview.save(path)


def create_rembg_session():
    """Use the same model cache and CPU provider during setup and slicing."""
    if not os.environ.get('U2NET_HOME') and not os.environ.get('REMBG_HOME'):
        os.environ['U2NET_HOME'] = str(Path(__file__).resolve().parents[1] / '.venv' / 'models')
    try:
        from rembg import new_session
    except ImportError as error:
        setup = Path(__file__).resolve().with_name('setup.sh')
        raise RuntimeError(f'rembg dependencies unavailable; run bash "{setup}"') from error
    return new_session('u2net', providers=['CPUExecutionProvider'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sheet')
    ap.add_argument('outdir')
    ap.add_argument('--mode', choices=['floodfill', 'rembg'], default='floodfill')
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--thresh', type=int, default=24, help='colour contrast guard for edge matting and QA, not a global erase threshold')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--count', type=int, default=None, help='real icons in row-major order; remaining cells are intentional blanks')
    a = ap.parse_args()
    if a.grid < 1 or a.size < 1 or not 1 <= a.thresh <= 255:
        ap.error('--grid and --size must be positive; --thresh must be between 1 and 255')
    total_cells = a.grid * a.grid
    if a.count is not None and not 1 <= a.count <= total_cells:
        ap.error(f'--count must be between 1 and {total_cells}')
    with Image.open(a.sheet) as source:
        im = source.convert('RGB')
    w, h = im.size
    if a.grid > min(w, h):
        ap.error('--grid cannot exceed the image dimensions')
    background = estimate_background(np.asarray(im), a.grid)
    mode = a.mode
    warnings = []
    if not background['confident']:
        warnings.append('Background estimate is uncertain; inspect source and previews. Colour-based QA is limited.')
        if mode == 'floodfill':
            mode = 'rembg'
            warnings.append('Used rembg because no reliable flat backdrop could be measured.')
    session = create_rembg_session() if mode == 'rembg' else None
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    qa_dir = outdir / 'qa'
    qa_dir.mkdir(exist_ok=True)
    reports, sources, cutouts, cleaned_images, outputs = [], [], [], [], []
    for index in range(a.count or total_cells):
        r, c = divmod(index, a.grid)
        x0, x1 = c * w // a.grid, (c + 1) * w // a.grid
        y0, y1 = r * h // a.grid, (r + 1) * h // a.grid
        pad = int(0.06 * min(x1 - x0, y1 - y0))
        px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
        px1, py1 = min(w, x1 + pad), min(h, y1 + pad)
        bounds = (x0 - px0, y0 - py0, x1 - px0, y1 - py0)
        cell = im.crop((px0, py0, px1, py1))
        rgb = np.asarray(cell)
        if mode == 'floodfill':
            alpha = floodfill_alpha(rgb, a.thresh, background['rgb'], background['tolerance'])
            cutout_rgb = rgb
        else:
            from rembg import remove
            result = np.asarray(remove(cell, session=session).convert('RGBA'))
            cutout_rgb, alpha = result[:, :, :3], result[:, :, 3]
        if background['confident']:
            cutout_rgb, alpha = refine_edges(
                cutout_rgb, alpha, background['rgb'], a.thresh, background['tolerance'], infer_alpha=mode == 'floodfill'
            )
        cleaned = clean(alpha, bounds)
        if cleaned is None:
            cleaned = np.zeros_like(alpha)
        report = inspect_cell(rgb, cutout_rgb, alpha, cleaned, bounds, background, a.thresh)
        rgba = Image.fromarray(np.dstack((cutout_rgb, cleaned)))
        output = normalise(rgba, a.size)
        filename = f'{index + 1:02d}.png'
        output.save(outdir / filename)
        report.update({'file': filename, 'cell': index + 1, 'source_bounds': [px0, py0, px1, py1], 'owned_bounds': list(bounds)})
        reports.append(report)
        sources.append(cell)
        cutouts.append(Image.fromarray(np.dstack((cutout_rgb, alpha))))
        cleaned_images.append(rgba)
        outputs.append(output)

    for name, images, colour in (
        ('source', sources, background['hex']),
        ('cutout', cutouts, '#ff00ff'),
        ('cleaned', cleaned_images, '#ff00ff'),
        ('magenta', outputs, '#ff00ff'),
        ('light', outputs, '#ffffff'),
        ('dark', outputs, '#202020'),
    ):
        save_preview(images, qa_dir / f'{name}.png', a.grid, colour, reports)
    review = [entry['cell'] for entry in reports if entry['findings']]
    report = {
        'sheet': str(Path(a.sheet).resolve()), 'requested_mode': a.mode, 'effective_mode': mode,
        'background': background, 'warnings': warnings, 'manual_review_required': True,
        'status': 'needs_review' if review or warnings else 'no_automatic_findings',
        'exported': len(reports), 'skipped_blank': total_cells - len(reports),
        'review_cells': review, 'cells': reports,
    }
    (qa_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'{a.sheet} -> {a.outdir}  mode={mode}  background={background["hex"]}  exported={len(reports)}/{total_cells}')
    print(f'QA: {qa_dir / "report.json"}  review_cells={review}; visual review required.')
    for warning in warnings:
        print(f'WARNING: {warning}')
    for entry in reports:
        if entry['findings']:
            print(f'{entry["file"]}: ' + ', '.join(item['code'] for item in entry['findings']))


if __name__ == '__main__':
    main()
