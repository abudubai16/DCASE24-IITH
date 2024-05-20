import torch
from torch import nn


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        print(module_out[1].shape)
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


d_model = 512
nhead = 8
dim_feedforward = 2048
dropout = 0.0
num_layers = 6

encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
transformer = nn.TransformerEncoder(encoder_layer, num_layers)

transformer.eval()

save_output = SaveOutput()
patch_attention(transformer.layers[-1].self_attn)
hook_handle = transformer.layers[-1].self_attn.register_forward_hook(save_output)

seq_len = 20
X = torch.rand(seq_len, 1, d_model)

with torch.no_grad():
    out = transformer(X)

print(save_output.outputs[0].shape)
