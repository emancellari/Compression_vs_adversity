# Image Compression versus Adversity

A single clean/adversarial pair analyzed with **JPEG** and **lossy WebP**.
The included test is **NbuGAN-0.90, ResNet50, ImageNet, acorn1 (ID:988)**.

## Files

- `jpeg_analysis.py`: JPEG entry script.
- `webp_analysis.py`: WebP entry script.
- `compression_analysis.py`: shared classification, compression and reporting.
- `examples/`: one clean image and its respective adversarial image.
- `requirements.txt`: dependencies.
- `appendix.pdf`: Supplementary documentation -> Figures 5 - 20 and Table 6.

## Run the included test

With Python 3.10 or 3.11, from this folder:

```bash
python -m pip install -r requirements.txt
python jpeg_analysis.py --ancestor acorn
python webp_analysis.py --ancestor acorn
```

Keras downloads ImageNet ResNet50 weights on the first run. Qualities are
100, 95, 90, 85, 80, 75, 70. Each encoding starts from original pixels and is
decoded in memory before classification.

Running creates `results/jpeg` and `results/webp`, containing top-five prediction
graphs, confidence curves, predictions/tracking CSVs, min/max/mean/population SD and run metadata. Use `--overwrite` to repeat a run.
Only the results for the included clean/adversarial pair are bundled in this folder, as test.

## Use any clean/adversarial pair from any attack

```bash
python jpeg_analysis.py --clean /path/to/clean.png \
  --adversarial /path/to/adversarial.png --ancestor source_label \
  --output-dir results/my_jpeg
python webp_analysis.py --clean /path/to/clean.png \
  --adversarial /path/to/adversarial.png --ancestor source_label \
  --output-dir results/my_webp
```

The images must correspond to each other. The fixed target defaults to the
original adversarial top-1; use `--target` to specify an intended class. The
ancestor defaults to the original clean top-1 if `--ancestor` is omitted.
The supplied pair is measured without attack-success filtering.

## Use any classifier trained on any dataset

Comments in `compression_analysis.py` mark where to replace the classifier,
labels and training preprocessing. Or pass `--adapter your_module:your_factory`
to either script. The factory returns `(predict, labels, metadata)`, where:

```python
# Here put any classifier trained on your dataset.
# Apply exactly the resize, color conversion and normalization used in training.
def your_factory():
    model = ...
    labels = [...]  # Unique class names, in the model's class-index order.

    def predict(bgr_pixels):
        # Here preprocess the image and run your classifier.
        probabilities = ...  # One-dimensional array, one probability per class.
        return probabilities

    return predict, labels, {"classifier": "your_model", "dataset": "your_dataset"}
```

Adapters receive OpenCV BGR pixels and must return finite probabilities in
[0,1] summing to 1. The built-in ResNet50 uses RGB, 224×224 Lanczos resize,
and Keras ResNet50 trained on ImageNet preprocessing.


