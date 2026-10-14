# Third-party notices

Original work in this repository is (c) Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0 (see `LICENSE`).
The components below are third-party and keep their own licenses.

## 1. FaceMe (code)

- Source: https://github.com/bnati5/FaceMe
- License: MIT
- Contains code from FaceMe (MIT license), used in the notebooks `2_training-face-mask-model.ipynb`, `3_detecting-mask-w-mtcnn.ipynb` and, to a smaller degree, `1_data-preprocessing.ipynb`.

```
MIT License

Copyright (c) 2020 Nathanael Alemayehu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Face Mask Detection dataset (data)

- Source: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection (author: andrewmvd)
- Original images and annotations: Make ML "Mask Dataset", https://makeml.app/datasets/mask
- License: CC0 1.0 Universal (public domain dedication), as stated on the Kaggle page.
- Where it applies: everything under `data/images/` (853 PNG photos) and `data/annotations/` (853 PASCAL VOC XML files), and the sample photo shown in the output of `1_data-preprocessing.ipynb`.
- Privacy note: the photos show real people's faces. CC0 covers the dataset as published; it says nothing about the consent of the people shown. If you appear in a photo and want it removed from this repository, contact the author through the portfolio link in `README.md`.

## 3. Runtime dependencies (not redistributed here)

`mtcnn` (MIT, https://github.com/ipazc/mtcnn), TensorFlow/Keras (Apache-2.0), OpenCV (Apache-2.0) and the ImageNet weights for InceptionV3 that Keras downloads on first use are installed or downloaded by the user and are not part of this repository.
