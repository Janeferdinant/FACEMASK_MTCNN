# Run log: Hugging Face Space check (2026-10-14)

**Result: the Space could not be created, so there is no live Space to test.** This log records what happened and a local rehearsal of the exact Space package. The rehearsal is **not** a test of a live Space.

## 1. Account and model repository

| Check | Result |
|---|---|
| `HfApi().whoami()` | logged in as user `JaneAaronNF`; token can write repos (fine-grained token with `repo.write`); account is not PRO (`isPro: False`) |
| Create public model repo `JaneAaronNF/face-mask-inceptionv3-mtcnn` | **succeeded**. Files: `README.md` (model card), `inceptionV3-model.keras` (188 MB), `best-model.weights.h5` (188 MB), `run_metrics.json`. https://huggingface.co/JaneAaronNF/face-mask-inceptionv3-mtcnn |

## 2. Creating the Space: failed

`python demo/build_space.py` called `HfApi.create_repo("JaneAaronNF/face-mask-demo", repo_type="space", space_sdk="gradio", private=False)`. Hugging Face answered:

```
HTTP 402 Payment Required for https://huggingface.co/api/repos/create
Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic
requires a PRO subscription. Subscribe at https://huggingface.co/pro
```

Nothing was created: `list_spaces(author="JaneAaronNF")` returns no Spaces. No other Hugging Face repository was touched, and no payment was attempted.

## 3. Local rehearsal of the Space package (not the live Space)

What it shows: the package that `demo/build_space.py` assembles (`app.py`, `README.md`, `requirements.txt`, `examples/`) starts, downloads the weights from the public model repository, and gives sensible output through the Gradio API. What it does not show: how the free Hugging Face CPU tier would build or run it (Linux image, 2 vCPU, cold starts). That remains untested.

| Item | Value |
|---|---|
| How | the folder assembled by `build_space.py` was run with `python app.py` on a **fresh Hugging Face cache**, so the weights really came from the model repository (`inceptionV3-model.keras`, 188 MB) |
| Environment | Windows 11 laptop CPU, Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1, Gradio 6.30.0 (the pins in `demo/space/requirements.txt`, apart from `opencv-python-headless`, which is the headless build of the same OpenCV version) |
| Start-up | download of the weights, model load and server start took less than 57 s in total (one check, so this is an upper bound) |
| Client | `gradio_client.Client(...).predict(handle_file(<example>), api_name="/run")`, three example photos |
| Linux wheels | `pip download` confirmed Linux (manylinux, cp311) wheels exist for `tensorflow==2.21.0`, `keras==3.15.1` and `opencv-python-headless==5.0.0.93`, so the pins can install on a Linux Space |

| Example photo (test split) | Time per request | Faces found | Labels |
|---|---|---|---|
| `maksssksksss3.png` | 4.5 s (includes warm-up) | 5 | 5 `with_mask` |
| `maksssksksss124.png` | 3.9 s | 6 | 2 `with_mask`, 4 `without_mask` |
| `maksssksksss250.png` | 1.8 s | 5 | 1 `with_mask`, 4 `without_mask` |

These are the same detections and labels as the round-1 local check on the `.h5` model (`--check` output), so the `.keras` file on the Hub is the same final model. As documented elsewhere, MTCNN found 5 of the 9 faces in photo 3, and photo 250 has no `mask_weared_incorrect` prediction. Face counts here are what the pipeline returned, not a ground-truth comparison.

Also checked: `python demo/app.py --check data/images/maksssksksss250.png` in a clone with no local weights downloads the weights from the model repository and prints 5 detections.

## 4. Alternatives, if a hosted demo is still wanted

1. **Subscribe to Hugging Face PRO** (the message above says that is what a free-account Gradio Space needs), then run `python demo/build_space.py`; the Space package is ready.
2. **Static Space** (free): no Python, so no TensorFlow. The model would have to be converted (for example to ONNX or TensorFlow.js) and MTCNN replaced by a JavaScript face detector, so images would stay in the visitor's browser. This is a separate piece of work and would be a different pipeline from the one in this repository.
3. **Other free hosting** for a Python app, or a Colab notebook that launches the same `demo/app.py`. Either needs an owner decision first.
