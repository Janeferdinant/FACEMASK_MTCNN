---
title: Face Mask Detection (MTCNN + InceptionV3)
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-nd-4.0
short_description: Detect faces with MTCNN and label mask / no mask / worn incorrectly
models:
  - JaneRoseFerdinant/face-mask-inceptionv3-mtcnn
---

# Face mask detection: MTCNN + InceptionV3

Upload a photo (or take a webcam snapshot) and the app draws a labelled box on every face it finds: `with_mask`, `without_mask` or `mask_weared_incorrect`. MTCNN finds the faces; a transfer-learned InceptionV3 classifier labels each crop.

This is a teaching demo of a two-stage detect-then-classify pipeline.

## Please read

- **Privacy.** Images are processed in memory and the app never stores them. Gradio briefly keeps an upload in a temporary cache, which this app clears every 10 minutes.
- **Examples.** The example images come only from the licensed dataset: Kaggle [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) by andrewmvd, CC0. It shows real people's faces; CC0 does not say whether they consented.
- **First load.** This Space runs on the free CPU tier. The first request after it has been idle can take about 30 seconds while it wakes up and loads the model.
- **Results are imperfect.**
  - MTCNN can miss faces: it found 5 of the 9 faces in one test photo, and a missed face gets no label.
  - The `mask_weared_incorrect` class has only 12 test crops in the evaluation and an F1 of 0.70 (Verified, re-run 2026-10-13, photo-grouped split; see the model card).
  - The classifier saw only one small dataset (853 photos), so photos with different lighting, angles, skin tones or mask styles will give more mistakes.
- **Not for safety-critical or surveillance use.** Do not use it to make decisions about people, to enforce rules, or to identify or track anyone.

## Links

- Model weights and model card: https://huggingface.co/JaneRoseFerdinant/face-mask-inceptionv3-mtcnn
- Code, notebooks, run log and project page: https://github.com/Janeferdinant/FACEMASK_MTCNN
- License: original work (c) Jane Rose Ferdinant, CC BY-NC-ND 4.0. Third-party components keep their own licenses; see THIRD_PARTY_NOTICES.md in the GitHub repository.

Jane Rose Ferdinant · [Portfolio](https://janeferdinant.vercel.app) · [LinkedIn](https://www.linkedin.com/in/janerosenf/)
