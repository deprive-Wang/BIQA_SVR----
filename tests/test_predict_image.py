import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from low_level import FEATURE_NAMES, LowLevelConfig, extract_low_level
from predict_image import predict_image


def test_predict_image_extracts_saved_feature_layout(tmp_path: Path) -> None:
    generator = np.random.default_rng(10)
    rgb = generator.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    image = tmp_path / 'input.png'
    Image.fromarray(rgb).save(image)
    vector = extract_low_level(rgb).reshape(1, -1)
    training = np.vstack((vector, vector + 1, vector + 2, vector + 3))
    model = make_pipeline(StandardScaler(), SVR(kernel='rbf')).fit(
        training, np.array([30.0, 40.0, 50.0, 60.0]),
    )
    run = tmp_path / 'run'
    model_dir = run / 'round_0000'
    model_dir.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({
        'feature_set': 'low', 'feature_names': list(FEATURE_NAMES),
        'feature_metadata': {'low_level': {'config': LowLevelConfig().metadata()}},
    }), encoding='utf-8')
    with (model_dir / 'model.pkl').open('wb') as stream:
        pickle.dump(model, stream)

    result = predict_image(image, run, device='cpu')

    assert result['feature_set'] == 'low'
    assert result['predicted_mos_raw'] == model.predict(vector)[0]
