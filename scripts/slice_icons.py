#!/usr/bin/env python3
"""Slice icon sheets, estimate their backdrop, and export PNGs with QA artifacts.

Flat art uses a measured colour key plus a narrow edge matte. Soft art uses
rembg. Components are assigned by cell ownership, including detached accents.
QA findings are review hints, never a claim of semantic correctness.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import shutil
import tempfile
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
            if (not isinstance(region, dict) or not {'seed', 'bounds', 'tolerance'} <= set(region)
                    or set(region) - {'seed', 'bounds', 'tolerance', 'refine_edge'}):
                raise ValueError(f'cell {cell}: each erase region needs seed, bounds and tolerance')
            if 'refine_edge' in region and type(region['refine_edge']) is not bool:
                raise ValueError(f'cell {cell}: refine_edge must be a boolean')
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


def erasure_mask(rgb, bg, region):
    x, y = region['seed']
    left, top, right, bottom = region['bounds']
    allowed = np.zeros(rgb.shape[:2], dtype=bool)
    allowed[top:bottom, left:right] = colour_distance(rgb[top:bottom, left:right], bg) <= region['tolerance']
    seed = np.zeros(allowed.shape, dtype=bool)
    seed[y, x] = True
    return ndimage.binary_propagation(seed, mask=allowed)


def erase_background_regions(rgb, alpha, bg, erasures):
    """Erase only the background-coloured region inside each rectangle."""
    result = alpha.copy()
    applied = []
    for region in erasures:
        erased = erasure_mask(rgb, bg, region)
        pixels = int(np.count_nonzero(erased & (result > 0)))
        result[erased] = 0
        applied.append({**region, 'removed_pixels': pixels})
    return result, applied


def apply_cutout_repairs(rgb, cutout_rgb, alpha, bg, erasures, thresh=24, tolerance=2, infer_alpha=True):
    """Optional edge unmixing is confined to a reviewed rectangle and 2px rim.

    Leave surrounding artwork and existing model alpha untouched by default.
    The returned mask identifies intentional edits for QA, not approval of the
    final image. The source RGB always determines the seeded erased region.
    """
    result, applied = erase_background_regions(rgb, alpha, bg, erasures)
    corrected = cutout_rgb.copy()
    intentional = np.zeros(alpha.shape, dtype=bool)
    masks = [erasure_mask(rgb, bg, region) for region in erasures]
    for mask in masks:
        intentional |= mask & (result == 0)
    for region, log, mask in zip(erasures, applied, masks):
        if not region.get('refine_edge', False):
            continue
        scope = ndimage.binary_dilation(mask, iterations=2) & ~mask
        scope &= cell_mask(alpha.shape, region['bounds']) & (result > 0)
        refined_rgb, refined_alpha = refine_edges(
            corrected, result, bg, thresh, tolerance, infer_alpha=infer_alpha,
        )
        changed = scope & ((refined_alpha != result) | np.any(refined_rgb != corrected, axis=2))
        corrected[changed] = refined_rgb[changed]
        result[changed] = refined_alpha[changed]
        intentional |= changed
        log['refined_edge_pixels'] = int(np.count_nonzero(changed))
    return corrected, result, applied, intentional


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


def inspect_cell(rgb, cutout_rgb, before, after, bounds, background, thresh, intentional=None):
    """Compare source, cutout, and cleanup; report ambiguities without erasing."""
    owned = cell_mask(after.shape, bounds)
    findings = []
    intentional = np.zeros(after.shape, dtype=bool) if intentional is None else intentional

    def flag(code, message, mask, minimum=3, severity='review'):
        objects = regions(mask, minimum)
        if objects:
            findings.append({
                'code': code, 'message': message, 'severity': severity,
                'pixels': sum(obj['area'] for obj in objects),
                'region_count': len(objects), 'regions': sorted(objects, key=lambda item: item['area'], reverse=True),
            })

    if not after.any():
        findings.append({'code': 'empty', 'severity': 'error', 'message': 'No foreground remains in this cell.'})
    removed = (before > 0) & (after == 0)
    labels, _ = ndimage.label(removed, structure=CONNECTED)
    # Only whole removed regions with no owned pixels are routine padding
    # cleanup. Keep the entire region under review if any part crosses inward.
    touches_owned = np.isin(labels, np.unique(labels[removed & owned]))
    flag('removed_components', 'Cleanup removed content overlapping this cell; check against the source.',
         removed & touches_owned, 1)
    flag('removed_padding', 'Cleanup removed fragments wholly outside this cell; retained as evidence.',
         removed & ~touches_owned, 1, severity='info')
    visible = after > 32
    holes = ndimage.binary_fill_holes(visible) & ~visible
    flag('transparent_holes', 'Enclosed transparency may be a valid hole or lost subject detail.', holes & owned & ~intentional)

    if background['confident']:
        bg, tolerance = background['rgb'], background['tolerance']
        similar = colour_distance(cutout_rgb, bg) <= max(thresh, tolerance)
        edge = (ndimage.distance_transform_edt(visible) <= 2) & visible
        edge_ambiguity = similar & edge & owned & ~intentional
        flag('background_like_foreground', 'Retained pixels resemble the backdrop; inspect for grey residue or intentional grey detail.', similar & visible & owned & ~edge & ~intentional)
        # Compare locally so a missing accent attached to a retained body is
        # reported even when both form one source-image component.
        source = (colour_distance(rgb, bg) > tolerance + 2) & owned
        missing = source & (before <= 32) & ~intentional
        near_edge = (ndimage.distance_transform_edt(~visible) <= 2) if visible.any() else np.zeros_like(visible)
        faint_loss = missing & near_edge & (colour_distance(rgb, bg) <= max(thresh, tolerance))
        edge_ambiguity |= faint_loss
        flag('edge_ambiguity', 'Low-contrast boundary differences; inspect antialiasing, grey fringe or thin details.', edge_ambiguity, severity='info')
        flag(
            'possible_cutout_loss',
            'Foreground-coloured source pixels are nearly transparent after cutout; inspect for lost details.',
            missing & ~faint_loss,
        )
    else:
        findings.append({
            'code': 'uncertain_background', 'severity': 'review',
            'message': 'Background sampling is unreliable; compare this cell with its source. Colour-based loss detection is unavailable.',
        })

    flag('repaired_background', 'Explicit local edits; inspect the resulting opening and rim.',
         intentional & owned, 1, severity='info')
    rim = owned & ~ndimage.binary_erosion(owned)
    flag('cell_boundary', 'Foreground reaches a cell boundary; check clipping or neighbour bleed.', visible & rim)
    return {
        'cutout_components': len(regions(before > 0)),
        'retained_components': len(regions(after > 0)),
        'removed_pixels': int(np.count_nonzero((before > 0) & (after == 0))),
        'findings': findings,
    }


FINDING_CODES = {
    'empty', 'removed_components', 'removed_padding', 'transparent_holes', 'background_like_foreground',
    'possible_cutout_loss', 'uncertain_background', 'cell_boundary',
    'candidate_disagreement', 'repaired_background', 'edge_ambiguity',
}
DISPOSITIONS = {'valid_hole', 'intentional_detail', 'edge_noise', 'verified_repair', 'false_positive', 'needs_fix'}


def load_review(path, count):
    """Validate annotations before exporting; old fingerprints remain auditable."""
    if path is None:
        return {}
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate review key: {key!r}')
            result[key] = value
        return result
    document = json.loads(Path(path).read_text(), object_pairs_hook=unique_keys)
    if not isinstance(document, dict) or set(document) != {'cells'} or not isinstance(document['cells'], dict):
        raise ValueError('review must contain only a cells object')
    result = {}
    for key, entry in document['cells'].items():
        if not key.isdecimal() or not 1 <= int(key) <= count or key != str(int(key)):
            raise ValueError(f'invalid review cell: {key!r}')
        if not isinstance(entry, dict) or set(entry) != {'fingerprint', 'status', 'reviewer', 'note', 'decisions'}:
            raise ValueError(f'review cell {key}: needs fingerprint, status, reviewer, note and decisions')
        fingerprint = entry['fingerprint']
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError(f'review cell {key}: invalid fingerprint')
        if entry['status'] not in ('accepted', 'needs_fix'):
            raise ValueError(f'review cell {key}: status must be accepted or needs_fix')
        if any(not isinstance(entry[field], str) or not entry[field].strip() for field in ('reviewer', 'note')):
            raise ValueError(f'review cell {key}: reviewer and note must be nonempty text')
        decisions = entry['decisions']
        if (not isinstance(decisions, dict) or set(decisions) - FINDING_CODES
                or any(not isinstance(value, str) or value not in DISPOSITIONS for value in decisions.values())):
            raise ValueError(f'review cell {key}: invalid finding decisions')
        if entry['status'] == 'accepted' and 'needs_fix' in decisions.values():
            raise ValueError(f'review cell {key}: cannot accept a cell that needs fixing')
        result[int(key)] = entry
    return result


def set_review_state(report, annotation):
    """Findings are immutable evidence; review is separate and version-bound."""
    review = {'status': 'pending'}
    if annotation is not None:
        review = dict(annotation)
        if annotation['fingerprint'] != report['fingerprint']:
            review.update(status='stale', reason='Source, processing or output changed; repeat visual review.')
        elif annotation['status'] == 'accepted':
            unresolved = [f['code'] for f in report['findings']
                          if f.get('severity', 'review') != 'info' and f['code'] not in annotation['decisions']]
            if unresolved:
                review.update(status='pending', reason='Missing decisions for: ' + ', '.join(unresolved))
    if any(f['code'] == 'empty' for f in report['findings']):
        review.update(status='needs_fix', reason='An empty exported icon cannot be accepted.')
    report['review'] = review
    report['automatic_status'] = ('error' if any(f.get('severity') == 'error' for f in report['findings'])
                                  else 'review' if any(f.get('severity', 'review') == 'review' for f in report['findings'])
                                  else 'info' if report['findings'] else 'clear')


def save_qa_report(qa_dir, report):
    """Keep the overview bounded and put all region coordinates in cell files."""
    details_dir = qa_dir / 'details'
    details_dir.mkdir(exist_ok=True)
    compact = json.loads(json.dumps(report))
    active = set()
    for entry in compact['cells']:
        filename = f'{entry["cell"]:02d}.json'
        active.add(filename)
        # One complete, compact JSON document per cell; no region is discarded.
        (details_dir / filename).write_text(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
        entry['details'] = str(Path('qa') / 'details' / filename)
        for finding in entry['findings']:
            objects = finding.pop('regions', [])
            finding['regions'] = objects[:3]
            if len(objects) > 3:
                finding['regions_in_details'] = len(objects) - 3
    for stale in details_dir.glob('*.json'):
        if stale.stem.isdecimal() and stale.name not in active:
            stale.unlink()
    # Keep one JSON entry on each line, including its bounded sample regions.
    lines = ['{']
    for key, value in compact.items():
        if key != 'cells':
            lines.append('  ' + json.dumps(key) + ': ' + json.dumps(value, ensure_ascii=False) + ',')
    lines.append('  "cells": [')
    lines.extend('    ' + json.dumps(entry, ensure_ascii=False) + (',' if i + 1 < len(compact['cells']) else '')
                 for i, entry in enumerate(compact['cells']))
    lines.extend(['  ]', '}'])
    (qa_dir / 'report.json').write_text('\n'.join(lines) + '\n')
    automatic = report['automatic_review_cells']
    unresolved = set(automatic) & set(report['review_cells'])
    summary = ['# Cutout QA', '', f'Status: **{report["status"]}** · accepted {len(report["accepted_cells"])}/{report["exported"]}',
               f'Awaiting visual acceptance: {len(report["review_cells"])} · needs fix: {len(report["needs_fix_cells"])}',
               f'Automatic review/error hints: {len(automatic)} cells · unresolved: {len(unresolved)} cells',
               '', 'Pending acceptance does not mean a detected defect. Automatic hints remain evidence after acceptance. Full regions: `details/NN.json`.', '',
               '| Cell | Review | Automatic | Findings / decisions |', '| --- | --- | --- | --- |']
    def priority(entry):
        state = entry['review']['status']
        return (0 if state == 'needs_fix' else 2 if state == 'accepted' else 1,
                {'error': 0, 'review': 1, 'info': 2, 'clear': 3}[entry['automatic_status']],
                state != 'stale', entry['cell'])

    for entry in sorted(compact['cells'], key=priority):
        decisions = entry['review'].get('decisions', {})
        findings = '; '.join(f['code'] + (': ' + decisions[f['code']] if f['code'] in decisions else '')
                             + (f' ({f["region_count"]} regions)' if 'region_count' in f else '')
                             for f in entry['findings']) or 'none'
        summary.append(f'| {entry["cell"]:02d} | {entry["review"]["status"]} | {entry["automatic_status"]} | {findings} |')
    if report['warnings']:
        summary += ['', 'Notes:'] + ['- ' + warning for warning in report['warnings']]
    (qa_dir / 'summary.md').write_text('\n'.join(summary) + '\n')


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
        state = reports[index]['review']['status']
        mark = {'accepted': 'OK', 'needs_fix': 'FIX', 'stale': 'STALE'}.get(state,
                'REVIEW' if reports[index]['automatic_status'] in ('review', 'error') else 'CHECK')
        colour = '#176b32' if state == 'accepted' else '#b22222' if mark in ('FIX', 'REVIEW', 'STALE') else '#555555'
        draw.text((x + 5, y + 5), f'{index + 1:02d} {mark}', fill=colour)
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


def previous_exports(destination):
    """Read only canonical export names from the previous QA manifest."""
    report_path = destination / 'qa' / 'report.json'
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text())
        if not isinstance(report, dict) or not isinstance(report.get('cells'), list):
            raise ValueError('expected a cells list')
        exports = {}
        for entry in report['cells']:
            if not isinstance(entry, dict):
                raise ValueError('invalid cell entry')
            cell, name = entry.get('cell'), entry.get('file')
            if (type(cell) is not int or cell < 1 or name != f'{cell:02d}.png'
                    or name in exports):
                raise ValueError('invalid or duplicate export filename')
            digest = entry.get('output_sha256')
            if 'output_sha256' in entry and (
                not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
            ):
                raise ValueError('invalid export hash')
            exports[name] = digest
        return exports
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f'Cannot safely read previous QA report {report_path}: {error}') from error


@contextmanager
def publication_lock(destination):
    """Serialize publication/recovery; keep the lock inode stable between runs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f'.{destination.name}.lock'
    if lock.is_symlink():
        raise ValueError(f'Refusing symlink publication lock: {lock}')
    with lock.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if handle.tell() == 0:
                handle.write(b' ')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def publication_fingerprint(path):
    """Hash node type and contents, including empty directories and symlinks."""
    if path.is_symlink():
        kind, content = 'symlink', os.fsencode(os.readlink(path))
    elif path.is_file():
        kind, content = 'file', path.read_bytes()
    elif path.is_dir():
        kind = 'directory'
        content = json.dumps([(child.name, publication_fingerprint(child))
                              for child in sorted(path.iterdir())], sort_keys=True).encode()
    elif not path.exists():
        return None
    else:
        raise ValueError(f'Unsupported publication file: {path}')
    return {'kind': kind, 'sha256': hashlib.sha256(content).hexdigest()}


def protected_input_paths(paths):
    """Protect every file/directory link traversed, plus the final target.

    Resolve each component before processing '..', including components inside
    link targets. Track only links being expanded so revisiting a directory
    link later in a finite path is not mistaken for a cycle.
    """
    protected = set()

    def resolve(path, expanding):
        resolved = Path(path.anchor)
        for part in path.parts[1:]:
            if part == '..':
                resolved = resolved.parent
                continue
            entry = resolved / part
            if entry.is_symlink():
                protected.add(entry)
                if entry in expanding:
                    raise ValueError(f'Symlink loop in input path: {entry}')
                target = Path(os.readlink(entry))
                if not target.is_absolute():
                    target = entry.parent / target
                resolved = resolve(target, expanding | {entry})
            else:
                resolved = entry
        return resolved

    for value in paths:
        if value is not None:
            protected.add(resolve(Path(value).absolute(), set()))
    return sorted(protected)


def input_path_relative_to(source, root):
    """Find an input's location under a root using actual ancestor identity.

    Path spelling alone misses case aliases on case-insensitive filesystems.
    Existing ancestors establish membership even if the input leaf is absent
    during recovery. Keep lexical matching for roots not yet restored, and do
    not lowercase paths: differently cased names can be distinct directories.
    """
    if source.is_relative_to(root):
        return source.relative_to(root)
    try:
        root_stat = root.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    for ancestor in (source, *source.parents):
        try:
            same = os.path.samestat(ancestor.stat(), root_stat)
        except (FileNotFoundError, NotADirectoryError):
            continue
        if same:
            return source.relative_to(ancestor)
    return None


def publication_inventory(root, names=None):
    """Record each file independently so interrupted cleanup can be resumed."""
    inventory = {}

    def visit(path):
        name = path.relative_to(root).as_posix()
        if path.is_dir() and not path.is_symlink():
            inventory[name] = {'kind': 'directory'}
            for child in sorted(path.iterdir()):
                visit(child)
        else:
            inventory[name] = publication_fingerprint(path)

    if root.exists():
        for path in sorted(root.iterdir()) if names is None else [root / name for name in names]:
            if path.exists() or path.is_symlink():
                visit(path)
    return inventory


def verify_cleanup(destination, record):
    """Allow already-deleted entries, but never erase added or edited data."""
    for key in ('stage', 'backup'):
        root = destination.parent / record[key]
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f'Unsafe cleanup directory: {root}')
        if record['version'] == 1:
            # Older journals have only whole-tree hashes. They are safe to
            # recover while each remaining tree is intact; otherwise stop.
            for path in root.iterdir():
                if key == 'backup' and path.name == 'transaction.next.json' and path.is_file() and not path.is_symlink():
                    continue
                if key == 'backup' and path.name == 'discarded' and path.is_dir() and not path.is_symlink():
                    entries = [(child, record['incoming'].get(child.name)) for child in path.iterdir()]
                else:
                    manifest = record['incoming'] if key == 'stage' else record['previous']
                    entries = [(path, manifest.get(path.name))]
                for child, expected in entries:
                    if expected is None or publication_fingerprint(child) != expected:
                        raise ValueError(f'Modified or untracked recovery file: {child}')
            continue
        actual = publication_inventory(root)
        if key == 'backup' and 'transaction.next.json' in actual:
            if actual['transaction.next.json']['kind'] != 'file':
                raise ValueError(f'Unsafe transaction temporary file: {root / "transaction.next.json"}')
            del actual['transaction.next.json']
        expected = record['cleanup'][key]
        for name, node in actual.items():
            if node != expected.get(name):
                raise ValueError(f'Modified or untracked recovery file: {root / name}')


def transaction_path(destination):
    return destination.parent / f'.{destination.name}-transaction.json'


def write_transaction(destination, record):
    # A process exit sees either the old complete record or the new complete
    # record. Save the first record before any output is moved.
    temporary = destination.parent / record['backup'] / 'transaction.next.json'
    with temporary.open('w') as handle:
        json.dump(record, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(transaction_path(destination))


def read_transaction(destination):
    path = transaction_path(destination)
    if not path.exists() and not path.is_symlink():
        return None
    try:
        if path.is_symlink():
            raise ValueError('journal is a symlink')
        record = json.loads(path.read_text())
        if (not isinstance(record, dict) or record.get('version') not in (1, 2)
                or not isinstance(record.get('destination'), str)
                or record.get('state') not in ('pending', 'committed', 'rolled_back')):
            raise ValueError('invalid transaction header')
        recorded_destination = Path(record['destination'])
        if not recorded_destination.is_absolute():
            raise ValueError('invalid transaction destination')
        if record['destination'] != str(destination):
            # resolve() preserves case aliases on case-insensitive filesystems.
            # Verify the output, sibling directory and journal identities before
            # trusting the original spelling of the temporary directory names.
            if (not recorded_destination.samefile(destination)
                    or not recorded_destination.parent.samefile(destination.parent)
                    or not transaction_path(recorded_destination).samefile(path)):
                raise ValueError('invalid transaction destination')
        for key in ('backup', 'stage'):
            name = record.get(key)
            prefix = f'.{recorded_destination.name}-{key}-'
            if (not isinstance(name, str) or not name.startswith(prefix)
                    or not name[len(prefix):] or Path(name).name != name
                    or '/' in name or '\\' in name):
                raise ValueError(f'invalid {key} directory')
            root = destination.parent / name
            if root.is_symlink() or (root.exists() and not root.is_dir()):
                raise ValueError(f'unsafe {key} directory')
        for key in ('previous', 'incoming'):
            entries = record.get(key)
            if not isinstance(entries, dict):
                raise ValueError(f'invalid {key} manifest')
            for name, node in entries.items():
                valid_name = (name == 'qa' or (name.endswith('.png')
                              and name[:-4].isascii() and name[:-4].isdecimal()
                              and int(name[:-4]) > 0 and name == f'{int(name[:-4]):02d}.png'))
                if not valid_name or not isinstance(node, dict):
                    raise ValueError('invalid transaction entry')
                digest = node.get('sha256')
                kinds = ('file',) if name != 'qa' else ('directory', 'symlink')
                if (node.get('kind') not in kinds or not isinstance(digest, str)
                        or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
                    raise ValueError('invalid transaction fingerprint')
        if 'qa' not in record['incoming']:
            raise ValueError('missing incoming QA')
        if record['version'] == 2:
            cleanup = record.get('cleanup')
            if not isinstance(cleanup, dict) or set(cleanup) != {'stage', 'backup'}:
                raise ValueError('invalid cleanup inventory')
            for key, entries in cleanup.items():
                if not isinstance(entries, dict):
                    raise ValueError('invalid cleanup inventory')
                for name, node in entries.items():
                    parts = name.split('/')
                    if (any(part in ('', '.', '..') for part in parts) or '\\' in name
                            or not isinstance(node, dict)):
                        raise ValueError('invalid cleanup entry')
                    allowed = record['incoming'] if key == 'stage' else record['previous']
                    if parts[0] not in allowed and not (key == 'backup' and parts[0] == 'discarded'):
                        raise ValueError('invalid cleanup root')
                    if (len(parts) > 1 and entries.get('/'.join(parts[:-1])) != {'kind': 'directory'}):
                        raise ValueError('missing cleanup parent directory')
                    digest = node.get('sha256')
                    if node != {'kind': 'directory'} and (
                        node.get('kind') not in ('file', 'symlink') or not isinstance(digest, str)
                        or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                    ):
                        raise ValueError('invalid cleanup fingerprint')
        return record
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f'Cannot safely recover publication from {path}: {error}') from error


def finish_transaction(destination, record):
    # Terminal state is recorded before recursive cleanup. Repeating cleanup
    # after a process exit must never roll back an already completed result.
    verify_cleanup(destination, record)
    for key in ('stage', 'backup'):
        root = destination.parent / record[key]
        if root.exists():
            shutil.rmtree(root)
    transaction_path(destination).unlink()


def recover_publication(destination, protected_paths=()):
    destination = Path(destination).resolve()
    with publication_lock(destination):
        _recover_publication(destination, protected_paths)


def _recover_publication(destination, protected_paths=()):
    record = read_transaction(destination)
    if record is None:
        return
    protected = protected_input_paths(protected_paths)
    for key in ('backup', 'stage'):
        root = destination.parent / record[key]
        for source in protected:
            if input_path_relative_to(source, root) is not None:
                raise FileExistsError(f'Recovery would remove input file: {source}; preserve it outside the temporary directory first.')
    if record['state'] != 'pending':
        finish_transaction(destination, record)
        return
    backup = destination.parent / record['backup']
    if not backup.is_dir():
        raise ValueError(f'Missing publication backup: {backup}')
    previous, incoming = record['previous'], record['incoming']
    names = sorted(previous.keys() | incoming.keys())
    current = {name: publication_fingerprint(destination / name) for name in names}
    saved = {name: publication_fingerprint(backup / name) for name in names}
    discarded_root = backup / 'discarded'
    if discarded_root.is_symlink() or (discarded_root.exists() and not discarded_root.is_dir()):
        raise ValueError(f'Unsafe recovery quarantine: {discarded_root}')
    remove = []
    # Validate the entire plan before touching any live file. A missing old
    # backup is acceptable only if that exact old node is already live.
    for name in names:
        before, after = previous.get(name), incoming.get(name)
        live, original = current[name], saved[name]
        if original is not None and (before is None or original != before):
            raise ValueError(f'Modified publication backup: {backup / name}')
        if before is not None and original is None:
            if live != before:
                raise ValueError(f'Cannot recover missing original: {destination / name}')
            continue
        if live is not None and live != after:
            raise ValueError(f'Refusing to overwrite modified recovery file: {destination / name}')
        if live is not None:
            target = destination / name
            for source in protected:
                relative = input_path_relative_to(source, target)
                if relative is not None:
                    restored = backup / name / relative
                    if publication_fingerprint(restored) != publication_fingerprint(source):
                        raise FileExistsError(f'Recovery would overwrite input file: {source}; preserve it outside the output directory first.')
            discarded = backup / 'discarded' / name
            if discarded.exists() or discarded.is_symlink():
                raise ValueError(f'Conflicting recovery file: {discarded}')
            remove.append(name)
    verify_cleanup(destination, record)
    # Withdraw QA before any PNG changes, quarantine new files by rename, and
    # restore the old QA last. Every step can be safely repeated after exit.
    for name in sorted(remove, key=lambda name: name != 'qa'):
        discarded = backup / 'discarded' / name
        discarded.parent.mkdir(exist_ok=True)
        (destination / name).replace(discarded)
    for name in sorted(names, key=lambda name: name == 'qa'):
        if saved[name] is not None:
            (backup / name).replace(destination / name)
    record['state'] = 'rolled_back'
    write_transaction(destination, record)
    finish_transaction(destination, record)


def publish_outputs(staged, destination, protected_paths=()):
    """Publish with a journal that can roll back a killed process on rerun."""
    destination = Path(destination).resolve()
    staged = Path(staged).resolve()
    with publication_lock(destination):
        _recover_publication(destination, protected_paths)
        _publish_outputs(staged, destination, protected_paths)


def _publish_outputs(staged, destination, protected_paths=(), *, qa_snapshot=...):
    """Publish numbered PNGs and QA together, withdrawing acceptance first.

    A failed publish restores PNGs before restoring the old QA. If rollback
    itself fails, the backup remains on disk and no old acceptance is exposed
    alongside partially replaced icons.
    """
    protected = protected_input_paths(protected_paths)

    def is_input(path):
        return any(path == source or path.resolve() == source or (path.exists() and source.exists() and path.samefile(source))
                   for source in protected)

    # A matching name alone never proves that a file is ours to remove.
    owned = set()
    for name, digest in previous_exports(destination).items():
        path = destination / name
        if path.is_symlink() or not path.is_file() or is_input(path):
            continue
        if digest is None or hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            owned.add(name)
    incoming = sorted(path for path in staged.glob('*.png') if path.stem.isdecimal()) + [staged / 'qa']
    for path in incoming[:-1]:
        target = destination / path.name
        if is_input(target):
            raise FileExistsError(f'Refusing to overwrite input file: {target}; use another output directory.')
        if (target.exists() or target.is_symlink()) and target.name not in owned:
            raise FileExistsError(f'Refusing to overwrite untracked or modified file: {target}; use another output directory.')
    # Inputs kept in QA must also survive its directory replacement unchanged.
    qa_path = destination / 'qa'
    for source in protected:
        relative = input_path_relative_to(source, qa_path)
        if relative is not None:
            copy = staged / 'qa' / relative
            if publication_fingerprint(copy) != publication_fingerprint(source):
                raise FileExistsError(f'Refusing to overwrite input file: {source}; use another output directory.')
    previous = [destination / 'qa'] + [destination / name for name in sorted(owned)]
    destination.mkdir(parents=True, exist_ok=True)
    old_manifest = {path.name: publication_fingerprint(path) for path in previous
                    if path.exists() or path.is_symlink()}
    if qa_snapshot is not ... and old_manifest.get('qa') != qa_snapshot:
        raise FileExistsError(f'QA changed during slicing: {qa_path}; current files were preserved. Rerun after finishing edits.')
    new_manifest = {path.name: publication_fingerprint(path) for path in incoming}
    stage_inventory = publication_inventory(staged)
    backup_inventory = publication_inventory(destination, old_manifest)
    backup_inventory['discarded'] = {'kind': 'directory'}
    backup_inventory.update({f'discarded/{name}': node for name, node in stage_inventory.items()})
    backup = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-backup-', dir=destination.parent))
    record = {
        'version': 2, 'destination': str(destination), 'state': 'pending',
        'backup': backup.name, 'stage': staged.name,
        'previous': old_manifest, 'incoming': new_manifest,
        'cleanup': {'stage': stage_inventory, 'backup': backup_inventory},
    }
    try:
        write_transaction(destination, record)
    except BaseException:
        # No output has been moved before the first complete journal exists.
        if not transaction_path(destination).exists():
            shutil.rmtree(backup)
        raise
    try:
        for path in previous:
            if path.name == 'qa' and publication_fingerprint(path) != old_manifest.get('qa'):
                raise FileExistsError(f'QA changed while publishing: {path}; current files and the transaction were preserved.')
            if path.exists() or path.is_symlink():
                path.replace(backup / path.name)
                if path.name == 'qa' and publication_fingerprint(backup / 'qa') != old_manifest['qa']:
                    # An editor need not obey our lock. Keep the changed backup
                    # and journal rather than committing and deleting its data.
                    raise FileExistsError(f'QA changed while publishing: {backup / "qa"}; preserve this backup before recovery.')
        for path in incoming:
            path.replace(destination / path.name)
        record['state'] = 'committed'
        write_transaction(destination, record)
    except BaseException:
        _recover_publication(destination, protected_paths)
        raise
    finish_transaction(destination, record)


def check_qa_output_links(qa_path):
    """Preserve user links, but never write generated QA through a link."""
    def reject_link(path):
        if path.is_symlink():
            raise FileExistsError(f'Refusing to replace a symlink at a generated QA path: {path}; preserve links and input files outside generated QA paths.')

    reject_link(qa_path)
    for name in ('report.json', 'summary.md', 'source.png', 'cutout.png',
                 'cleaned.png', 'magenta.png', 'light.png', 'dark.png', 'colour-candidate.png'):
        reject_link(qa_path / name)
    for directory, suffix in (('details', '.json'), ('colour-candidates', '.png')):
        parent = qa_path / directory
        reject_link(parent)
        if parent.is_dir():
            for child in parent.iterdir():
                # Also guard uppercase aliases on case-insensitive volumes.
                if child.suffix.lower() == suffix and child.stem.isdecimal():
                    reject_link(child)


@contextmanager
def staged_output(destination, protected_paths=()):
    """Keep the QA snapshot and publication in the same exclusive run."""
    with publication_lock(destination):
        _recover_publication(destination, protected_paths)
        qa_path = destination / 'qa'
        check_qa_output_links(qa_path)
        qa_snapshot = publication_fingerprint(qa_path)
        staged = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-stage-', dir=destination.parent))
        try:
            if qa_snapshot is not None:
                # Copy link nodes, including dangling links, without visiting
                # their targets. Both trees keep the same relative path depth.
                shutil.copytree(qa_path, staged / 'qa', symlinks=True)
                check_qa_output_links(staged / 'qa')
                if (publication_fingerprint(staged / 'qa') != qa_snapshot
                        or publication_fingerprint(qa_path) != qa_snapshot):
                    raise FileExistsError(f'QA changed while copying: {qa_path}; current files were preserved. Rerun after finishing edits.')
            yield staged
            _publish_outputs(staged, destination, protected_paths, qa_snapshot=qa_snapshot)
        finally:
            # A failed recovery keeps its stage together with the journal and
            # backup, so a later invocation can inspect the complete evidence.
            if not transaction_path(destination).exists() and staged.exists():
                shutil.rmtree(staged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sheet')
    ap.add_argument('outdir')
    ap.add_argument('--mode', choices=['floodfill', 'rembg'], default='floodfill')
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--thresh', type=int, default=24, help='colour contrast guard for edge matting and QA, not a global erase threshold')
    ap.add_argument('--background-tolerance', type=int, default=None, help='override only border-connected background removal, from 0 to 255')
    ap.add_argument('--repairs', type=Path, help='agent-reviewed per-cell modes, tolerances and bounded background erasures (JSON)')
    ap.add_argument('--review-file', type=Path, help='version-bound visual review decisions (JSON); rerun with the same slicing settings')
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
    destination = Path(a.outdir).resolve()
    protected_paths = (a.sheet, a.repairs, a.review_file)
    # Hold the same lock during recovery, input reads, processing and publication.
    with staged_output(destination, protected_paths) as outdir:
        with Image.open(a.sheet) as source:
            im = source.convert('RGB')
        w, h = im.size
        if a.grid > min(w, h):
            ap.error('--grid cannot exceed the image dimensions')
        background = estimate_background(np.asarray(im), a.grid)
        count = a.count or total_cells
        try:
            repairs = load_repairs(a.repairs, im, a.grid, count, background['rgb'])
            annotations = load_review(a.review_file, count)
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
        qa_dir = outdir / 'qa'
        qa_dir.mkdir(exist_ok=True)
        if uses_rembg:
            (qa_dir / 'colour-candidates').mkdir(exist_ok=True)
        reports, sources, cutouts, cleaned_images, outputs, colour_candidates = [], [], [], [], [], []
        run_context = {
            'source_sha256': hashlib.sha256(im.tobytes()).hexdigest(),
            'processor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'grid': a.grid, 'count': count, 'size': a.size, 'thresh': a.thresh, 'background': background,
            'requested_mode': a.mode, 'background_tolerance': a.background_tolerance,
        }
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
            # Local edits leave surrounding art intact unless a bounded rim was
            # explicitly selected for colour unmixing after source comparison.
            colour_rgb, colour_alpha, colour_repairs, intentional = apply_cutout_repairs(
                rgb, colour_rgb, colour_alpha, background['rgb'], erasures, a.thresh, background['tolerance'],
            )
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
                cutout_rgb, alpha, applied_repairs, intentional = apply_cutout_repairs(
                    rgb, cutout_rgb, alpha, background['rgb'], erasures, a.thresh, background['tolerance'], infer_alpha=False,
                )
            cleaned = clean(alpha, bounds)
            if cleaned is None:
                cleaned = np.zeros_like(alpha)
            report = inspect_cell(rgb, cutout_rgb, alpha, cleaned, bounds, background, a.thresh, intentional)
            rgba = Image.fromarray(np.dstack((cutout_rgb, cleaned)))
            output = normalise(rgba, a.size)
            if output.getchannel('A').getbbox() is None and not any(f['code'] == 'empty' for f in report['findings']):
                report['findings'].append({
                    'code': 'empty', 'severity': 'error',
                    'message': 'No foreground remains in the final exported PNG after resizing.',
                })
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
                        'code': 'candidate_disagreement', 'severity': 'review',
                        'message': 'Colour candidate retains pixels removed by rembg; compare with source for lost detail or residual background.',
                        'pixels': sum(obj['area'] for obj in objects),
                        'region_count': len(objects), 'regions': sorted(objects, key=lambda item: item['area'], reverse=True),
                    })
            else:
                candidate = rgba
            digest = hashlib.sha256(json.dumps({'context': run_context, 'cell': report}, sort_keys=True).encode())
            for image in (output, candidate):
                digest.update(image.tobytes())
            digest.update(cutout_rgb.tobytes())
            digest.update(alpha.tobytes())
            digest.update(cleaned.tobytes())
            report['fingerprint'] = digest.hexdigest()
            report['output_sha256'] = hashlib.sha256((outdir / filename).read_bytes()).hexdigest()
            set_review_state(report, annotations.get(index + 1))
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
        review = [entry['cell'] for entry in reports if entry['review']['status'] != 'accepted']
        accepted = [entry['cell'] for entry in reports if entry['review']['status'] == 'accepted']
        needs_fix = [entry['cell'] for entry in reports if entry['review']['status'] == 'needs_fix']
        automatic_review = [entry['cell'] for entry in reports if entry['automatic_status'] in ('review', 'error')]
        report = {
            'schema_version': 2,
            'sheet': str(Path(a.sheet).resolve()), 'requested_mode': a.mode, 'effective_mode': mode,
            'background': background, 'warnings': warnings, 'manual_review_required': bool(review),
            'status': 'needs_fix' if needs_fix else 'needs_review' if review else 'accepted',
            'accepted_cells': accepted, 'needs_fix_cells': needs_fix,
            'exported': len(reports), 'skipped_blank': total_cells - len(reports),
            'review_cells': review, 'automatic_review_cells': automatic_review, 'cells': reports,
        }
        save_qa_report(qa_dir, report)
    print(f'{a.sheet} -> {a.outdir}  mode={mode}  background={background["hex"]}  exported={len(reports)}/{total_cells}')
    print(f'QA: {destination / "qa" / "summary.md"}  status={report["status"]}  '
          f'awaiting_acceptance={len(review)}  automatic_flags={len(automatic_review)}  '
          f'unresolved_flags={len(set(review) & set(automatic_review))}  needs_fix={len(needs_fix)}')
    for warning in warnings:
        print(f'WARNING: {warning}')
    for entry in reports:
        state = entry['review']['status']
        if state == 'accepted':
            continue
        findings = [item['code'] for item in entry['findings'] if item['severity'] in ('review', 'error')]
        if findings or state in ('needs_fix', 'stale'):
            print(f'{entry["file"]} [{state}/{entry["automatic_status"]}]: ' +
                  (', '.join(findings) or entry['review'].get('reason', 'See visual review note in report.json.')))


if __name__ == '__main__':
    main()
