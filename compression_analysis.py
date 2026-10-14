"""Paired-image compression measurements; no compressed images are saved."""
import argparse
import csv
import hashlib
import importlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np

QUALITIES = (100, 95, 90, 85, 80, 75, 70)
BASE = Path(__file__).resolve().parent


# CLASSIFIER: the demo uses ImageNet ResNet50.
# Here you can use any classifier trained on any dataset: replace this factory,
# including its labels, input resize, preprocessing and probability prediction.
# Alternatively pass --adapter your_module:your_factory.
def resnet50_adapter():
    """Return a BGR-array probability function, ordered labels, and model metadata."""
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.applications.imagenet_utils import decode_predictions
    import tensorflow as tf
    model = ResNet50(weights='imagenet')
    decoded = decode_predictions(np.arange(1000, dtype='float32')[None, :], top=1000)[0]
    labels = [''] * 1000
    for _, label, index in decoded:
        labels[int(index)] = label

    # ImageNet has repeated display names (for example crane); retain class identity.
    repeated = {name for name in labels if labels.count(name) > 1}
    labels = [f'{name} [class {i}]' if name in repeated else name for i, name in enumerate(labels)]

    def predict(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        batch = preprocess_input(resized.astype('float32')[None, ...])
        return model.predict(batch, verbose=0)[0]

    return predict, labels, dict(classifier='ResNet50', dataset='ImageNet-1k',
                                weights='imagenet', tensorflow_version=tf.__version__,
                                resize='224x224, OpenCV INTER_LANCZOS4',
                                preprocessing='RGB to Keras ResNet50 caffe preprocessing')


def validate_probabilities(values, class_count):
    values = np.asarray(values, dtype='float64')
    if values.shape != (class_count,):
        raise ValueError(f'Adapter must return shape ({class_count},), got {values.shape}')
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Adapter must return finite class probabilities in [0, 1]')
    if not np.isclose(values.sum(), 1, atol=1e-4):
        raise ValueError('Class probabilities must sum to 1; convert logits to probabilities')
    return values


def compress_decode(pixels, codec, quality):
    """Encode each quality from original pixels, then decode at source resolution."""
    if codec not in ('jpeg', 'webp') or not 0 <= quality <= 100:
        raise ValueError('Expected jpeg/webp and quality in [0,100]')
    extension, flag = ('.jpg', cv2.IMWRITE_JPEG_QUALITY) if codec == 'jpeg' else ('.webp', cv2.IMWRITE_WEBP_QUALITY)
    ok, encoded = cv2.imencode(extension, pixels, [flag, quality])
    if not ok:
        raise ValueError(f'{codec} encoding failed at quality {quality}')
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f'{codec} decoding failed at quality {quality}')
    return decoded, len(encoded)


def estimate_crossings(target, ancestor, qualities=QUALITIES):
    events = []
    delta = np.asarray(target) - np.asarray(ancestor)
    for i, q in enumerate(qualities):
        if delta[i] == 0:
            events.append(dict(status='equal_at_sample', quality_high=q, quality_low=q,
                               estimated_quality=q, confidence_percent=target[i]))
        if i + 1 < len(qualities) and delta[i] * delta[i + 1] < 0:
            fraction = delta[i] / (delta[i] - delta[i + 1])
            events.append(dict(status='interpolated_crossing', quality_high=q,
                               quality_low=qualities[i + 1],
                               estimated_quality=q + fraction * (qualities[i + 1] - q),
                               confidence_percent=target[i] + fraction * (target[i + 1] - target[i])))
    return events


def write_csv(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_reports(output, codec, labels, samples, target, ancestor):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    codec_label = 'JPEG' if codec == 'jpeg' else 'WebP'
    for kind, series in samples.items():
        fig, (baseline, ax) = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={'width_ratios': [1, 3]})
        original = series[0]['probabilities']
        top = np.argsort(-original, kind='stable')[:5]
        baseline.barh(range(len(top)), original[top] * 100, color='#636363')
        baseline.set_yticks(range(len(top)))
        baseline.set_yticklabels([labels[i].replace('_', ' ') for i in top])
        baseline.invert_yaxis()
        baseline.set_xlim(0, 110)
        baseline.set_xlabel('Confidence (%)')
        baseline.set_title('Original image: top 5')
        for row, i in enumerate(top):
            baseline.text(original[i] * 100 + 1, row, f'{original[i]:.2%}', va='center')
        used = list(dict.fromkeys(int(i) for s in series[1:] for i in np.argsort(-s['probabilities'], kind='stable')[:5]))
        colors = {i: plt.get_cmap('tab20')(n % 20) for n, i in enumerate(used)}
        for column, sample in enumerate(series[1:]):
            probs = sample['probabilities']
            best = np.argsort(-probs, kind='stable')[:5]
            label_positions, previous = [], -5
            for i in reversed(best):
                previous = max(probs[i] * 100, previous + 6)
                label_positions.append(previous)
            for i, y in zip(best, reversed(label_positions)):
                ax.scatter(column, probs[i] * 100, color=colors[int(i)], s=30, zorder=4)
                ax.annotate(f'{labels[i].replace("_", " ")}\n{probs[i]:.2%}',
                            xy=(column, probs[i]*100), xytext=(column+.07, y), fontsize=7,
                            color=colors[int(i)], va='center',
                            arrowprops=dict(arrowstyle='-', color=colors[int(i)], lw=.5))
        for index, color, role in ((target, '#b2182b', 'Fixed target'), (ancestor, '#2166ac', 'Ancestor')):
            ax.plot(range(len(QUALITIES)), [s['probabilities'][index]*100 for s in series[1:]],
                    'o-', color=color, lw=2, label=f'{role}: {labels[index]}')
        ax.scatter(0, original[target]*100, facecolors='none', edgecolors='#8c510a',
                   s=80, lw=1.8, label='Original target baseline', zorder=6)
        ax.set_xticks(range(len(QUALITIES)))
        ax.set_xticklabels(QUALITIES)
        ax.set_xlim(-.2, len(QUALITIES)-.05)
        ax.set_ylim(-2, 115)
        ax.set_xlabel(f'{codec_label} quality (100 → 70)')
        ax.set_ylabel('Confidence (%)')
        ax.set_title(f'{codec_label} compression: top 5 and fixed classes')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.suptitle(f'{kind}: {series[0]["source_image"]}')
        fig.tight_layout()
        fig.savefig(output / f'{kind}_analysis.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 8))
    for kind, index, color, role in (('adversarial', target, '#b2182b', 'Adversarial target'),
                                     ('adversarial', ancestor, '#2166ac', 'Adversarial ancestor'),
                                     ('clean', ancestor, '#238b45', 'Clean ancestor')):
        values = [s['probabilities'][index]*100 for s in samples[kind][1:]]
        ax.plot(QUALITIES, values, 'o-', color=color, lw=2, label=role)
        for q, value in zip(QUALITIES, values):
            ax.annotate(f'{value:.1f}%', (q, value), xytext=(0, 9), textcoords='offset points',
                        ha='center', fontsize=8, color=color,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.4))
    ax.scatter(100, samples['adversarial'][0]['probabilities'][target]*100, facecolors='none',
               edgecolors='#8c510a', s=100, lw=2, label='Original adversarial target', zorder=6)
    ax.scatter(100, samples['adversarial'][1]['probabilities'][target]*100, color='#8c510a',
               s=35, label=f'{codec_label}100 target', zorder=7)
    ax.set_xlim(102, 68)
    ax.set_ylim(-3, 108)
    ax.set_xticks(QUALITIES)
    ax.set_xlabel(f'{codec_label} quality (100 → 70)')
    ax.set_ylabel('Confidence (%)')
    ax.set_title(f'Compression vs adversity — {codec_label}')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2, fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'confidence_overview.png', dpi=180)
    plt.close(fig)


def analyze(clean_path, adversarial_path, output, codec, predict, labels, metadata,
            ancestor_label=None, target_label=None, overwrite=False):
    if codec not in ('jpeg', 'webp'):
        raise ValueError('codec must be jpeg or webp')
    labels = list(labels)
    if len(labels) < 2 or len(set(labels)) != len(labels) or not all(isinstance(s, str) and s for s in labels):
        raise ValueError('Adapter labels must be distinct, nonempty strings in class-index order')
    paths = dict(clean=Path(clean_path), adversarial=Path(adversarial_path))
    pixels = {}
    for kind, path in paths.items():
        pixels[kind] = cv2.imread(str(path))
        if pixels[kind] is None:
            raise ValueError(f'Could not read {kind} image: {path}')
    original = {kind: validate_probabilities(predict(image), len(labels)) for kind, image in pixels.items()}
    if ancestor_label is not None and ancestor_label not in labels:
        raise ValueError(f'Unknown ancestor label: {ancestor_label}')
    if target_label is not None and target_label not in labels:
        raise ValueError(f'Unknown target label: {target_label}')
    ancestor = labels.index(ancestor_label) if ancestor_label is not None else int(original['clean'].argmax())
    target = labels.index(target_label) if target_label is not None else int(original['adversarial'].argmax())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=overwrite)
    samples, tracking, predictions, statistics, events = {}, [], [], [], []
    for kind, image in pixels.items():
        samples[kind] = []
        for quality in (None,) + QUALITIES:
            if quality is None:
                probs, byte_count = original[kind], paths[kind].stat().st_size
            else:
                decoded, byte_count = compress_decode(image, codec, quality)
                probs = validate_probabilities(predict(decoded), len(labels))
            common = dict(kind=kind, source_image=paths[kind].name,
                          variant='original' if quality is None else 'recompressed',
                          quality='' if quality is None else quality,
                          format=paths[kind].suffix[1:].upper() if quality is None else codec.upper(),
                          encoded_bytes=byte_count)
            samples[kind].append(dict(common, probabilities=probs))
            top1 = int(probs.argmax())
            tracking.append(dict(common, target_label=labels[target], ancestor_label=labels[ancestor],
                                 target_confidence_percent=probs[target]*100,
                                 ancestor_confidence_percent=probs[ancestor]*100,
                                 difference_percentage_points=(probs[target]-probs[ancestor])*100,
                                 top1_label=labels[top1], ancestor_is_top1=top1 == ancestor))
            for rank, index in enumerate(np.argsort(-probs, kind='stable')[:5], 1):
                predictions.append(dict(common, rank=rank, class_index=int(index), label=labels[index],
                                        confidence_percent=probs[index]*100))
        for index, role in ((target, 'target'), (ancestor, 'ancestor')):
            for variant, chosen in (('original', samples[kind][:1]), ('recompressed', samples[kind][1:])):
                values = np.array([s['probabilities'][index]*100 for s in chosen])
                statistics.append(dict(kind=kind, series=role, label=labels[index], variant=variant,
                                       statistics_scope='across_measured_compression_qualities' if variant == 'recompressed' else 'single_original',
                                       sample_count=len(values), average_percent=values.mean(), min_percent=values.min(),
                                       max_percent=values.max(), standard_deviation_pp=values.std(ddof=0)))
        ct = [s['probabilities'][target]*100 for s in samples[kind][1:]]
        ca = [s['probabilities'][ancestor]*100 for s in samples[kind][1:]]
        points = estimate_crossings(ct, ca) if target != ancestor else []
        points = points or [dict(status='same_category' if target == ancestor else 'no_crossing_observed',
                                quality_high='', quality_low='', estimated_quality='', confidence_percent='')]
        events.extend(dict(kind=kind, **point) for point in points)
    means = []
    for i, quality in enumerate((None,) + QUALITIES):
        row = dict(variant='original' if quality is None else 'recompressed', quality='' if quality is None else quality, image_count=1)
        for key, kind, index in (('ct', 'adversarial', target), ('ca', 'adversarial', ancestor), ('clean_ca', 'clean', ancestor)):
            value = samples[kind][i]['probabilities'][index]*100
            for stat in ('mean', 'min', 'max'):
                row[f'{stat}_{key}_percent'] = value
            row[f'std_{key}_percent'] = 0.0
        row['difference_percentage_points'] = row['mean_ct_percent'] - row['mean_ca_percent']
        means.append(row)
    for filename, rows in (('predictions.csv', predictions), ('tracking.csv', tracking),
                           ('confidence_statistics.csv', statistics), ('crossings.csv', events), ('all_means.csv', means)):
        write_csv(output / filename, rows)
    lines = ['# Confidence statistics', '',
             'Recompressed rows summarize the seven measured quality levels (population SD, ddof=0).',
             'They do not describe variability across images. Original rows have one sample.', '',
             '| Image | Class | Variant | Samples | Mean (%) | Min (%) | Max (%) | SD (pp) |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in statistics:
        lines.append(f'| {r["kind"]} | {r["series"]} | {r["variant"]} | {r["sample_count"]} | {r["average_percent"]:.2f} | {r["min_percent"]:.2f} | {r["max_percent"]:.2f} | {r["standard_deviation_pp"]:.2f} |')
    (output / 'confidence_statistics.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    render_reports(output, codec, labels, samples, target, ancestor)
    manifest = dict(codec=codec, qualities=list(QUALITIES), model=metadata,
                    python_version=platform.python_version(), numpy_version=np.__version__,
                    opencv_version=cv2.__version__,
                    inputs={kind: dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                       shape=list(pixels[kind].shape)) for kind, path in paths.items()},
                    target_label=labels[target], ancestor_label=labels[ancestor],
                    target_selection='explicit_label' if target_label is not None else 'original_adversarial_top1',
                    ancestor_selection='explicit_label' if ancestor_label is not None else 'original_clean_top1',
                    screening='none: the supplied pair is always measured',
                    encoded_images_saved=False)
    (output / 'run.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'{codec.upper()}: target={labels[target]}, ancestor={labels[ancestor]}; saved reports to {output}')


def main(codec):
    parser = argparse.ArgumentParser(description=f'Analyze one clean/adversarial pair using {codec.upper()}.')
    # PAIR: put any clean image and its respective adversarial image from any attack here.
    # The defaults below are the single supplied NbuGAN-0.90 / ResNet50 test pair.
    # Override them using --clean and --adversarial for your own experiment.
    parser.add_argument('--clean', type=Path, default=BASE / 'examples/clean_acorn1.jpeg')
    parser.add_argument('--adversarial', type=Path, default=BASE / 'examples/r50_adv_acorn1_0.90.png')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'results' / codec)
    parser.add_argument('--ancestor', help='Dataset class label; default: original clean top-1')
    parser.add_argument('--target', help='Fixed class label; default: original adversarial top-1')
    parser.add_argument('--adapter', help='Custom model factory as module:function (default: ImageNet ResNet50)')
    parser.add_argument('--overwrite', action='store_true', help='Replace reports in an existing output directory')
    args = parser.parse_args()
    if args.output_dir.exists() and not args.overwrite:
        parser.error('Output exists; choose --output-dir or use --overwrite')
    try:
        if args.adapter:
            module, function = args.adapter.rsplit(':', 1)
            predict, labels, metadata = getattr(importlib.import_module(module), function)()
        else:
            # Replace with your classifier factory if you are not using --adapter.
            predict, labels, metadata = resnet50_adapter()
        analyze(args.clean, args.adversarial, args.output_dir, codec, predict, labels, metadata,
                args.ancestor, args.target, args.overwrite)
    except (ValueError, OSError, cv2.error) as error:
        parser.error(str(error))
