import matplotlib.pyplot as plt
import torch
import torchaudio

from dcase24t6.nn.hub_baseline import baseline_pipeline


def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure(figsize=(20, 6), dpi=80)
    ax = fig.add_subplot(111)
    ax.matshow(attentions.cpu().numpy(), cmap="gray_r")
    # fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(range(attentions.shape[-1]))
    ax.set_yticks(range(attentions.shape[-2]))
    ax.set_xticklabels(
        [item * 0.3125 for item in range(1, attentions.shape[-1] + 1)], rotation=90
    )
    ax.set_yticklabels(output_words + ["<EOS>"])

    # Show label at every tick
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("foo.png")
    # plt.show()


# Getting attention maps
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


model = baseline_pipeline()
model.eval()

save_output = SaveOutput()
patch_attention(model[1].decoder.layers[-1].multihead_attn)
hook_handle = (
    model[1].decoder.layers[-1].multihead_attn.register_forward_hook(save_output)
)

# print(model[1].decoder.layers[-1].self_attn)

# Forward Pass
audio_path = "/home/akhil/models/DCASE24/dcase2024-task6-baseline/data/CLOTHO_v2.1/clotho_audio_files/evaluation/Collingwood bees, bumble bees.wav"
audio_path = "/home/akhil/Baleno_Diesel_Engine_Sound.wav"
# audio_path = "/home/akhil/models/DCASE24/dcase2024-task6-baseline/data/CLOTHO_v2.1/clotho_audio_files/evaluation/20080504.horse.drawn.00.wav"
# audio_path = "/home/akhil/models/DCASE24/dcase2024-task6-baseline/data/CLOTHO_v2.1/clotho_audio_files/evaluation/20061215.early.morning.wav"

audio, sr = torchaudio.load(audio_path)
item = {"audio": audio, "sr": sr}

with torch.no_grad():
    outputs = model(item)

candidate = outputs["candidates"][0]

print("Caption : ", candidate)
print("Attention Weights : ", save_output.outputs[-1].shape)

showAttention(None, candidate.split(" "), save_output.outputs[-1][0, :, :])
