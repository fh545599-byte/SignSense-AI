# Dataset

SignSense AI is trained and evaluated on **GTSRB (German Traffic Sign Recognition Benchmark)**.

- **Link:** https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- **License:** CC0-1.0 (public domain)
- **Size:** about 612 MB, so it is not stored in this repository (GitHub limits files to 100 MB)
- **Content:** 43 traffic sign classes, 39,209 training images and 12,630 official test images

## Download

Manual: open the link above, click **Download**, and extract the zip into a folder named `gtsrb/`.

With the Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d gtsrb
```

## Expected folder layout

```
gtsrb/
├── Train/        training images, one folder per class (0-42)
├── Train.csv     image path and class of every training image
├── Test/         official test images
├── Test.csv      image path and class of every test image
├── Meta/         reference image of each sign
└── Meta.csv
```

## Use it with this project

Evaluate the trained model on the official test set:
```bash
python evaluate.py --data gtsrb
```

The model was trained with the extra class "Not a traffic sign" using CIFAR-100 photos
(downloaded automatically by Keras) and generated sign-like images. The generator code
is inside `notebook/SignSense_Train_v2.ipynb`.
