#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort

from musicfm.model.musicfm_25hz import MusicFM25Hz


MODEL_PATH = "./musicfm/data/pretrained_msd.pt"
STAT_PATH = "./musicfm/data/msd_stats.json"
ONNX_PATH = "musicfm_embedding.onnx"

SAMPLE_RATE = 24000
SEGMENT_SECONDS = 30
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS


DEVICE = "cpu"


# ---------------------------------------------------------------------
# PYTORCH
# ---------------------------------------------------------------------

def load_pytorch():
    model = MusicFM25Hz(
        is_flash=False,
        stat_path=STAT_PATH,
        model_path=MODEL_PATH,
    ).eval().to(DEVICE)

    return model


def pytorch_embedding(model, audio):
    with torch.no_grad():
        latent = model.get_latent(audio, layer_ix=7)

        pooled = latent.sum(dim=1) / latent.shape[1]
        pooled = F.normalize(pooled, dim=-1)

    return pooled.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------

def load_onnx():
    return ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"],
    )


def onnx_embedding(sess, audio):
    x = audio.cpu().numpy().astype(np.float32)

    input_name = sess.get_inputs()[0].name

    out = sess.run(None, {input_name: x})[0]

    return out.squeeze(0)


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def max_diff(a, b):
    return float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------
# RUN TEST
# ---------------------------------------------------------------------

def run_test():
    print("Loading models...")

    pt_model = load_pytorch()
    ort_sess = load_onnx()

    # IMPORTANT: 30s segment (real usecase)
    torch.manual_seed(0)
    audio = torch.randn(1, SEGMENT_SAMPLES, dtype=torch.float32)

    print("Running PyTorch...")
    pt_out = pytorch_embedding(pt_model, audio)

    print("Running ONNX...")
    onnx_out = onnx_embedding(ort_sess, audio)

    print("\n=== Results ===")
    print("Cosine similarity:", cosine(pt_out, onnx_out))
    print("Max abs diff     :", max_diff(pt_out, onnx_out))


    # thresholds tuned for embedding systems
    if cosine(pt_out, onnx_out) > 0.9999:
        print("\n✅ PASS: embeddings are effectively identical")
    elif cosine(pt_out, onnx_out) > 0.999:
        print("\n⚠️ WARNING: small drift but acceptable")
    else:
        print("\n❌ FAIL: ONNX divergence too large")


if __name__ == "__main__":
    run_test()
