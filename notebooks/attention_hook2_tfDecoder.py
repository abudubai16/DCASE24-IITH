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
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


d_model = 512
nhead = 8
dim_feedforward = 2048
dropout = 0.0
num_layers = 6

decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

memory = torch.rand(10, 32, 256)
tgt = torch.rand(20, 32, 256)

save_output = SaveOutput()
patch_attention(transformer_decoder.layers[-1].multihead_attn)
hook_handle = transformer_decoder.layers[-1].multihead_attn.register_forward_hook(
    save_output
)

with torch.no_grad():
    out = transformer_decoder(tgt, memory)

print(save_output.outputs[0].shape)
