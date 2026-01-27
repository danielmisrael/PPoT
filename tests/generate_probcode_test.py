import sys
sys.path.append("..")

import pytest
from transformers import AutoTokenizer
import ppot.compile
import torch
import numpy as np


def test_get_token_pos():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    code1 = '1 2 34 4 poorva(56789'
    token_ids = torch.tensor([tokenizer(code1)['input_ids']])
    a = ppot.compile.get_token_pos(token_ids, tokenizer)
    assert (a[0] == np.array([0, 2, 4, 5, 7, 11, 12, 13, 14, 15])).all()

    code1 = '>= <= == > <'
    token_ids = torch.tensor([tokenizer(code1)['input_ids']])
    a = ppot.compile.get_token_pos(token_ids, tokenizer, rule=r"[<>]=?|==")
    print(a)
    assert (a[0] == np.array([0, 1, 2, 3, 4])).all()

    code1 = "poorva"
    token_ids = torch.tensor([tokenizer(code1)['input_ids']])
    a = ppot.compile.get_token_pos(token_ids, tokenizer)
    assert (a[0] == np.array([])).all()
    
test_get_token_pos()
