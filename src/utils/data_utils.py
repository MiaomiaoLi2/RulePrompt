import json
import torch

# IO
def load_data(text_path, encoding='utf-8'):
    """
    load textual data from file
    """
    with open(text_path, encoding=encoding) as fp:
        texts = fp.readlines()
    return [t.strip() for t in texts]

# def load_json(path):
#     with open(path, 'r', encoding='utf8') as fh:
#         content = json.load(fh)
#     return content

# def save_json(path, content):
#     with open(path, 'w', encoding='utf8') as fh:
#         json.dump(content, fh, indent=4)

def to_cuda(inputs, is_tensor=True, device="cuda"):
    if isinstance(inputs, list):
        if not is_tensor:
            inputs = [torch.tensor(e) for e in inputs]
        return [e.to(device) for e in inputs]
    elif isinstance(inputs, dict):
        for e in inputs:
            if not is_tensor:
                inputs[e] = torch.tensor(inputs[e])
            inputs[e] = inputs[e].to(device)
    else:
        if not is_tensor:
            inputs = torch.tensor(inputs)
        inputs = inputs.to(device)
    return inputs

