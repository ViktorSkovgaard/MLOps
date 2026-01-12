import torch
import pytest
import re

from my_project.model import MyAwesomeModel

def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)


def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match=re.escape('Expected input to a 4D tensor')):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match=re.escape('Expected each sample to have shape [1, 28, 28]')):
        model(torch.randn(1,1,28,29))