Converts the musicfm sonic similarity model to an onnx runtime.

Inside of this repo's dir, run

```
wget -P musicfm/data/ https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json
wget -P musicfm/data/ https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt
```

Then install the dependencies with the flake or using `requirements.txt` and run

```
python export_onnx.py
```

Then it will output the onnx embedding model with its weights. A copy of this is available in the release.
