"""Predict one image's raw MOS score with a saved BCQI SVR model."""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from low_level import FEATURE_NAMES, LowLevelConfig, extract_low_level
from semantic import (CROP_SIZE, SEMANTIC_NAMES, WEIGHTS_SHA256,
                      WEIGHT_TRANSFORM, SemanticExtractor)


def _read_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f'Image does not exist: {path}')
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != 'RGB':
                raise ValueError(f'Expected RGB image, received {image.mode}')
            return np.asarray(image).copy()
    except OSError as error:
        raise ValueError(f'Cannot decode image {path}: {error}') from error


def predict_image(image_path: Path, run: Path, *, round_index: int = 0,
                  device: str = 'cuda') -> dict:
    """Return an uncalibrated MOS prediction from a trusted saved run."""
    if type(round_index) is not int or round_index < 0:
        raise ValueError('round_index must be a nonnegative integer')
    if device not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda')
    run = run.resolve()
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Saved config must be a JSON object')
    feature_set = config.get('feature_set')
    columns = {
        'combined': FEATURE_NAMES + SEMANTIC_NAMES,
        'low': FEATURE_NAMES,
        'semantic': SEMANTIC_NAMES,
    }
    if feature_set not in columns or config.get('feature_names') != list(columns[feature_set]):
        raise ValueError('Saved feature names do not match this extractor')
    metadata = config.get('feature_metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('Saved feature metadata must be a JSON object')
    if feature_set in {'combined', 'low'}:
        if metadata.get('low_level', {}).get('config') != LowLevelConfig().metadata():
            raise ValueError('Saved low-level preprocessing differs from current code')
    if feature_set in {'combined', 'semantic'}:
        semantic = metadata.get('semantic', {})
        if (semantic.get('architecture') != 'squeezenet1_1'
                or semantic.get('weights_sha256') != WEIGHTS_SHA256
                or semantic.get('crop_size') != CROP_SIZE
                or semantic.get('mean') != list(WEIGHT_TRANSFORM.mean)
                or semantic.get('std') != list(WEIGHT_TRANSFORM.std)
                or semantic.get('resize') is not False):
            raise ValueError('Saved semantic preprocessing differs from current code')
    model_path = run / f'round_{round_index:04d}' / 'model.pkl'
    if not model_path.is_file():
        raise ValueError(f'Saved model is missing: {model_path}')
    rgb = _read_rgb(image_path)
    features = []
    if feature_set in {'combined', 'low'}:
        features.append(extract_low_level(rgb))
    if feature_set in {'combined', 'semantic'}:
        features.append(SemanticExtractor(device=device).extract([rgb])[0])
    vector = np.concatenate(features).reshape(1, -1)
    if vector.shape[1] != len(columns[feature_set]) or not np.isfinite(vector).all():
        raise ValueError('Image features are incomplete or non-finite')
    # Pickle is only safe for models produced and trusted by this project.
    with model_path.open('rb') as stream:
        model = pickle.load(stream)
    if getattr(model, 'n_features_in_', None) != vector.shape[1]:
        raise ValueError('Saved model and feature dimensions disagree')
    prediction = float(model.predict(vector)[0])
    if not np.isfinite(prediction):
        raise ValueError('Model produced a non-finite prediction')
    return {
        'image': str(image_path.resolve()), 'model': str(model_path),
        'feature_set': feature_set, 'predicted_mos_raw': prediction,
        'score_kind': 'raw SVR prediction; no test-set logistic mapping',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--round', type=int, default=0, dest='round_index')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    try:
        result = predict_image(args.image, args.run, round_index=args.round_index,
                               device=args.device)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'Prediction failed: {error}\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
