import pytest
import torch
import numpy as np
from ml.model import build_model, AudioClassifier


def test_model_4_blocks_output_shape():
    model = build_model(num_classes=20, num_conv_blocks=4)
    x = torch.randn(4, 1, 64, 157)
    out = model(x)
    assert out.shape == (4, 20)


def test_model_3_blocks_output_shape():
    model = build_model(num_classes=20, num_conv_blocks=3)
    x = torch.randn(4, 1, 64, 157)
    out = model(x)
    assert out.shape == (4, 20)


def test_model_different_n_mels():
    for n_mels in [32, 64, 128]:
        model = build_model(num_classes=20, n_mels=n_mels)
        time_steps = 80000 // 512 + 1
        x = torch.randn(2, 1, n_mels, time_steps)
        out = model(x)
        assert out.shape == (2, 20)


def test_model_has_adaptive_avg_pool():
    model = build_model(num_classes=20, num_conv_blocks=4)
    last_block = list(model.features.children())[-1]
    has_adaptive = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in last_block.children())
    assert has_adaptive, "Last block must have AdaptiveAvgPool2d"


def test_model_last_block_no_maxpool():
    model = build_model(num_classes=20, num_conv_blocks=4)
    last_block = list(model.features.children())[-1]
    has_maxpool = any(isinstance(m, torch.nn.MaxPool2d) for m in last_block.children())
    assert not has_maxpool, "Last block must NOT have MaxPool2d"


def test_model_classifier_structure():
    model = build_model(num_classes=20)
    children = list(model.classifier.children())
    types = [type(c).__name__ for c in children]
    assert types == ['Flatten', 'Linear', 'ReLU', 'Dropout', 'Linear']


def test_model_batch_size_1():
    model = build_model(num_classes=20)
    model.eval()
    with torch.inference_mode():
        out = model(torch.randn(1, 1, 64, 157))
    assert out.shape == (1, 20)
