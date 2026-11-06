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


def border_connected(mask):
    seeds = np.zeros(mask.shape, dtype=bool)
    seeds[[0, -1], :] = mask[[0, -1], :]
    seeds[:, [0, -1]] = mask[:, [0, -1]]
    return ndimage.binary_propagation(seeds, mask=mask)


def floodfill_alpha(rgb, thresh=24, bg=None, tolerance=2, background_tolerance=None):
    """Remove measured background, preserving merely similar interior colours.

    Border-connected noise uses the measured or explicitly reviewed tolerance;
    enclosed regions require an almost exact match.
    Exact backdrop-coloured subject details cannot be distinguished from holes
    by colour alone, so QA reports every substantial enclosed transparent area.
    An explicit background_tolerance controls only border-connected removal;
    it does not widen enclosed erasure or change the measured background noise.
    """
    if bg is None:
        background = estimate_background(rgb)
        bg, tolerance = background['rgb'], background['tolerance']
    diff = colour_distance(rgb, bg)
    exterior_tolerance = tolerance if background_tolerance is None else background_tolerance
    exterior = border_connected(diff <= exterior_tolerance)
    # A deliberately narrow exterior threshold must not turn retained backdrop
    # noise or near-grey accents into "holes". Use the measured noise envelope
    # to identify outside regions, without erasing them at that wider tolerance.
    outside = exterior if exterior_tolerance >= tolerance else border_connected(diff <= tolerance)
    exact_holes = (diff <= min(1, tolerance)) & ~outside
    return np.where(exterior | exact_holes, 0, 255).astype(np.uint8)


def cell_geometry(index, width, height, grid):
    row, column = divmod(index, grid)
    x0, x1 = column * width // grid, (column + 1) * width // grid
    y0, y1 = row * height // grid, (row + 1) * height // grid
    pad = int(0.06 * min(x1 - x0, y1 - y0))
    left, top = max(0, x0 - pad), max(0, y0 - pad)
    right, bottom = min(width, x1 + pad), min(height, y1 + pad)
    return (left, top, right, bottom), (x0 - left, y0 - top, x1 - left, y1 - top)


def load_repairs(path, image, grid, count, bg):
    """Validate agent-reviewed per-cell overrides before exporting anything.

    Seeds and half-open bounds use the expanded source-cell coordinates from
    QA, and must stay inside that cell's owned_bounds. Never alter neighbours.
    """
    if path is None:
        return {}
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate repair key: {key!r}')
            result[key] = value
        return result

    document = json.loads(Path(path).read_text(), object_pairs_hook=unique_keys)
    if not isinstance(document, dict) or set(document) != {'cells'} or not isinstance(document['cells'], dict):
        raise ValueError('repairs must contain only a "cells" object keyed by 1-based cell numbers')

    def integer(value, low, high):
        return type(value) is int and low <= value <= high

    repairs = {}
    for key, settings in document['cells'].items():
        if not key.isdecimal() or not 1 <= int(key) <= count or key != str(int(key)):
            raise ValueError(f'invalid repair cell: {key!r}')
        cell = int(key)
        if not isinstance(settings, dict) or set(settings) - {'mode', 'background_tolerance', 'erase_regions'}:
            raise ValueError(f'cell {cell}: unsupported repair settings')
        if 'mode' in settings and settings['mode'] not in ('floodfill', 'rembg'):
            raise ValueError(f'cell {cell}: mode must be floodfill or rembg')
        if 'background_tolerance' in settings and not integer(settings['background_tolerance'], 0, 255):
            raise ValueError(f'cell {cell}: background_tolerance must be an integer from 0 to 255')
        erasures = settings.get('erase_regions', [])
        if not isinstance(erasures, list):
            raise ValueError(f'cell {cell}: erase_regions must be a list')
        crop, owned = cell_geometry(cell - 1, *image.size, grid)
        rgb = np.asarray(image.crop(crop))
        for region in erasures:
            if not isinstance(region, dict) or set(region) != {'seed', 'bounds', 'tolerance'}:
                raise ValueError(f'cell {cell}: each erase region needs seed, bounds and tolerance')
            seed, bounds, tolerance = region['seed'], region['bounds'], region['tolerance']
            if (not isinstance(seed, list) or len(seed) != 2
                    or not isinstance(bounds, list) or len(bounds) != 4
                    or not all(type(value) is int for value in seed + bounds)
                    or not integer(tolerance, 0, 255)):
                raise ValueError(f'cell {cell}: invalid erase region coordinates or tolerance')
            x, y = seed
            left, top, right, bottom = bounds
            if not (owned[0] <= left <= x < right <= owned[2]
                    and owned[1] <= top <= y < bottom <= owned[3]):
                raise ValueError(f'cell {cell}: erase seed and bounds must be inside owned_bounds')
            if np.max(np.abs(rgb[y, x].astype(float) - bg)) > tolerance:
                raise ValueError(f'cell {cell}: erase seed does not match the background at this tolerance')
        repairs[cell] = settings
    return repairs


def erase_background_regions(rgb, alpha, bg, erasures):
    """Erase only the selected background-coloured region inside each rectangle."""
    result = alpha.copy()
    if not erasures:
        return result, []
    distance = colour_distance(rgb, bg)
    applied = []
    for region in erasures:
        x, y = region['seed']
        left, top, right, bottom = region['bounds']
        allowed = np.zeros(alpha.shape, dtype=bool)
        allowed[top:bottom, left:right] = distance[top:bottom, left:right] <= region['tolerance']
        seed = np.zeros(alpha.shape, dtype=bool)
        seed[y, x] = True
        erased = ndimage.binary_propagation(seed, mask=allowed)
        pixels = int(np.count_nonzero(erased & (result > 0)))
        result[erased] = 0
        applied.append({**region, 'removed_pixels': pixels})
    return result, applied


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
    else:
        findings.append({
            'code': 'uncertain_background',
            'message': 'Background sampling is unreliable; compare this cell with its source. Colour-based loss detection is unavailable.',
        })

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
    ap.add_argument('--background-tolerance', type=int, default=None, help='override only border-connected background removal, from 0 to 255')
    ap.add_argument('--repairs', type=Path, help='agent-reviewed per-cell modes, tolerances and bounded background erasures (JSON)')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--count', type=int, default=None, help='real icons in row-major order; remaining cells are intentional blanks')
    a = ap.parse_args()
    if a.grid < 1 or a.size < 1 or not 1 <= a.thresh <= 255:
        ap.error('--grid and --size must be positive; --thresh must be between 1 and 255')
    if a.background_tolerance is not None and not 0 <= a.background_tolerance <= 255:
        ap.error('--background-tolerance must be between 0 and 255')
    total_cells = a.grid * a.grid
    if a.count is not None and not 1 <= a.count <= total_cells:
        ap.error(f'--count must be between 1 and {total_cells}')
    with Image.open(a.sheet) as source:
        im = source.convert('RGB')
    w, h = im.size
    if a.grid > min(w, h):
        ap.error('--grid cannot exceed the image dimensions')
    background = estimate_background(np.asarray(im), a.grid)
    count = a.count or total_cells
    try:
        repairs = load_repairs(a.repairs, im, a.grid, count, background['rgb'])
    except (OSError, ValueError) as error:
        ap.error(str(error))
    default_mode = a.mode
    warnings = []
    if not background['confident']:
        warnings.append('Background estimate is uncertain; inspect source and previews. Colour-based QA is limited.')
        if default_mode == 'floodfill':
            default_mode = 'rembg'
    modes = [repairs.get(i + 1, {}).get('mode', default_mode) for i in range(count)]
    mode = modes[0] if len(set(modes)) == 1 else 'mixed'
    uses_rembg = 'rembg' in modes
    if uses_rembg:
        warnings.append('rembg can remove subject details; compare each affected cell with its source and saved colour candidate.')
    session = create_rembg_session() if uses_rembg else None
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    qa_dir = outdir / 'qa'
    qa_dir.mkdir(exist_ok=True)
    if uses_rembg:
        (qa_dir / 'colour-candidates').mkdir(exist_ok=True)
    reports, sources, cutouts, cleaned_images, outputs, colour_candidates = [], [], [], [], [], []
    for index in range(count):
        crop, bounds = cell_geometry(index, w, h, a.grid)
        cell = im.crop(crop)
        rgb = np.asarray(cell)
        settings = repairs.get(index + 1, {})
        cell_mode = modes[index]
        exterior_tolerance = settings.get('background_tolerance', a.background_tolerance)
        if exterior_tolerance is None:
            exterior_tolerance = background['tolerance']
        erasures = settings.get('erase_regions', [])
        colour_alpha = floodfill_alpha(
            rgb, a.thresh, background['rgb'], background['tolerance'], exterior_tolerance,
        )
        colour_rgb = rgb
        if background['confident']:
            colour_rgb, colour_alpha = refine_edges(
                rgb, colour_alpha, background['rgb'], a.thresh, background['tolerance'],
            )
        # Finish edge inference first: a newly erased hole must not cause
        # neighbouring opaque artwork to be reinterpreted as a blended edge.
        colour_alpha, colour_repairs = erase_background_regions(rgb, colour_alpha, background['rgb'], erasures)
        if cell_mode == 'floodfill':
            cutout_rgb, alpha, applied_repairs = colour_rgb, colour_alpha, colour_repairs
        else:
            from rembg import remove
            result = np.asarray(remove(cell, session=session).convert('RGBA'))
            cutout_rgb, alpha = result[:, :, :3], result[:, :, 3]
            if background['confident']:
                cutout_rgb, alpha = refine_edges(
                    cutout_rgb, alpha, background['rgb'], a.thresh, background['tolerance'], infer_alpha=False,
                )
            alpha, applied_repairs = erase_background_regions(rgb, alpha, background['rgb'], erasures)
        cleaned = clean(alpha, bounds)
        if cleaned is None:
            cleaned = np.zeros_like(alpha)
        report = inspect_cell(rgb, cutout_rgb, alpha, cleaned, bounds, background, a.thresh)
        rgba = Image.fromarray(np.dstack((cutout_rgb, cleaned)))
        output = normalise(rgba, a.size)
        filename = f'{index + 1:02d}.png'
        output.save(outdir / filename)
        report.update({
            'file': filename, 'cell': index + 1, 'source_bounds': list(crop), 'owned_bounds': list(bounds),
            'mode': cell_mode, 'background_tolerance': exterior_tolerance, 'applied_repairs': applied_repairs,
        })
        if cell_mode == 'rembg':
            colour_cleaned = clean(colour_alpha, bounds)
            if colour_cleaned is None:
                colour_cleaned = np.zeros_like(colour_alpha)
            candidate = Image.fromarray(np.dstack((colour_rgb, colour_cleaned)))
            candidate_path = Path('qa') / 'colour-candidates' / filename
            candidate.save(outdir / candidate_path)
            report['colour_candidate'] = str(candidate_path)
            disagreement = (colour_cleaned > 32) & (cleaned <= 32) & cell_mask(cleaned.shape, bounds)
            objects = regions(disagreement, minimum=3)
            if objects:
                report['findings'].append({
                    'code': 'candidate_disagreement',
                    'message': 'Colour candidate retains pixels removed by rembg; compare with source for lost detail or residual background.',
                    'pixels': sum(obj['area'] for obj in objects),
                    'region_count': len(objects), 'regions': objects[:50],
                })
        else:
            candidate = rgba
        reports.append(report)
        sources.append(cell)
        cutouts.append(Image.fromarray(np.dstack((cutout_rgb, alpha))))
        cleaned_images.append(rgba)
        outputs.append(output)
        colour_candidates.append(candidate)
    for name, images, colour in (
        ('source', sources, background['hex']),
        ('cutout', cutouts, '#ff00ff'),
        ('cleaned', cleaned_images, '#ff00ff'),
        ('magenta', outputs, '#ff00ff'),
        ('light', outputs, '#ffffff'),
        ('dark', outputs, '#202020'),
    ):
        save_preview(images, qa_dir / f'{name}.png', a.grid, colour, reports)
    if uses_rembg:
        save_preview(colour_candidates, qa_dir / 'colour-candidate.png', a.grid, '#ff00ff', reports)
    else:
        (qa_dir / 'colour-candidate.png').unlink(missing_ok=True)
    active_candidates = {entry['file'] for entry in reports if 'colour_candidate' in entry}
    for stale in (qa_dir / 'colour-candidates').glob('*.png'):
        if stale.stem.isdecimal() and stale.name not in active_candidates:
            stale.unlink()
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
