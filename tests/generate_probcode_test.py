import sys
sys.path.append("..")

import pytest
from transformers import AutoTokenizer
import genPPS


def test_get_tokens_pos():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    code1 = '1 2 34 4 poorva(56789'
    assert genPPS.generate_probcode.get_tokens_pos(code1, tokenizer) == [0, 2, 4, 5, 7, 11, 12, 13, 14, 15]
