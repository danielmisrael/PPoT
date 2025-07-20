import torch
import re

def extract_code(response_str):
    """Extract code from response string"""
    matches = re.findall(r'```python(.*?)```', response_str, re.DOTALL)
    if matches:
        return "\n".join(match.strip() for match in matches)
    else:
        return response_str
    
    
# TODO we should agree on the formatting of the probabilities dictionary.

def compile_probcode(code, probabilities):
    '''
    Inputs:
        code: str
        probabilities: dict[str, float] Maps strings contained within code (constants or variables) to their probabilities
    Returns:
        probabilistic_code: str Still a string, but includes source of randomness and incorporates probabilities.
    
    '''
    # TODO @Poorva
    pass


def get_probs(token_ids, logits):
    '''
    Inputs:
        token_ids: list[int]
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
    Returns:
        probabilities: dict[str, float] Maps strings contained within code (constants or variables) to their probabilities
    '''
    # TODO @Renato
    pass

    
# Sketch of what the generation loop will look like
def genenerate_probcode(model, tokenizer, input_ids, **gen_kwargs):
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids, 
            **gen_kwargs,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    code = extract_code(output_text[0])
    
    code_ids = tokenizer.encode(code)
        
    with torch.no_grad():
        logits = model(generated_ids).logits
        
        
    code_logits = logits[:, -len(code_ids):, :]
    
    probabilities = get_probs(code_ids, code_logits)
    
    probabilistic_code = compile_probcode(code, probabilities)
    
    return probabilistic_code
        
        




        
        
