#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import onnx

from musicfm.model.musicfm_25hz import MusicFM25Hz


MODEL_PATH = "./musicfm/data/pretrained_msd.pt"
STAT_PATH = "./musicfm/data/msd_stats.json"
ONNX_PATH = "musicfm_embedding.onnx"

SAMPLE_RATE = 24000
SEGMENT_SECONDS = 30
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

DEVICE = "cpu"


# ---------------------------------------------------------------------
# Determinism (important for reproducibility)
# ---------------------------------------------------------------------

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------
# MODEL WRAPPER (stable embedding extraction)
# ---------------------------------------------------------------------

class MusicFMEmbedding(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # freeze model weights (reduces graph/runtime variance)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, audio):
        # latent: [B, T, D] or similar
        latent = self.model.get_latent(audio, layer_ix=7)

        # stable pooling (avoid torch.mean for reproducibility)
        pooled = torch.sum(latent, dim=1) / latent.shape[1]

        # normalize INSIDE graph (critical for stable similarity space)
        pooled = F.normalize(pooled, dim=-1)

        return pooled


# ---------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------

def export_onnx():
    print("Loading model...")

    model = MusicFM25Hz(
        is_flash=False,
        stat_path=STAT_PATH,
        model_path=MODEL_PATH,
    ).to(DEVICE).eval().float()

    wrapper = MusicFMEmbedding(model).eval().float()

    dummy = torch.randn(
        1,
        SEGMENT_SAMPLES,
        dtype=torch.float32,
        device=DEVICE,
    )

    print("Exporting ONNX...")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            ONNX_PATH,
            dynamo=True,
        )

    print("Validating ONNX...")

    model_onnx = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model_onnx)

    print("Saved:", ONNX_PATH)


# ---------------------------------------------------------------------

if __name__ == "__main__":
    export_onnx()
