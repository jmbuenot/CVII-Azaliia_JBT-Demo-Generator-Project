# Pix2Pix Variants — MoNuSeg Inference Demo

This program  runs **inference only** on histopathology images using several trained Pix2Pix generator variants and produces side-by-side comparison plots.

This is the demo program for **Task 3** of the Generative Models project.
## Available trained models

| Name       | Description                                                       | Architecture                |
|------------|-------------------------------------------------------------------|-----------------------------|
| `baseline` | Standard Pix2Pix (BCE adversarial + 100·L1 pixel loss)            | U-Net                       |
| `lsgan`    | LSGAN adversarial loss + L1 pixel loss                            | U-Net                       |
| `bce_l2`   | BCE adversarial + L2 (MSE) pixel loss                             | U-Net                       |
| `l1_only`  | Pure L1 regression — no discriminator (ablation)                  | U-Net                       |
| `augmented`| Standard Pix2Pix loss + paired data augmentation                  | U-Net                       |
| `resnet`   | ResNet-based generator (CycleGAN-style, 9 residual blocks)        | ResNet                      |
| `attunet`  | U-Net with attention gates on skip connections                    | Attention U-Net             |
| `transfer` | U-Net with ResNet34 encoder pre-trained on ImageNet               | smp.Unet (ResNet34/ImageNet)|

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/jmbuenot/CVII-Azaliia_JBT-Demo-Generator-Project.git
cd <your-repo>

# 2. Create a virtual environment (that step is optional)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `segmentation-models-pytorch` is only required for the `transfer`
> model. If you do not install it, the script will skip the `transfer` model
> automatically and continue with the rest.

## Project layout

```
demo_inference/
├── inference.py          # the demo program
├── README.md
├── requirements.txt
├── .gitignore
├── models/               # trained generator checkpoints (.pth)
│   ├── best_generator.pth
│   ├── best_generator_lsgan_l1.pth
│   ├── best_generator_bce_l2.pth
│   ├── best_generator_l1_only.pth
│   ├── best_generator_augmented.pth
│   ├── best_generator_resnet.pth
│   ├── best_generator_attunet.pth
│   └── best_generator_transfer_resnet34.pth
├── samples/              # sample input images (MoNuSeg or others)
│   ├── sample_001.png
│   └── ...
└── outputs/              # created at runtime, holds comparison plots
```

## Usage

### Run inference on a single MoNuSeg paired image

Use the `--paired` flag so the script keeps only the
input half and also displays the ground truth in the comparison plot:

```bash
python inference.py --input samples/sample_001.png --paired
```

### Run inference on a whole directory of MoNuSeg images

```bash
python inference.py --input samples/ --paired --output outputs/
```

### Run inference on a non-paired (plain) input image

 We also have regular tissue images, so drop the
`--paired` flag:

```bash
python inference.py --input samples/not_paired/my_tissue1.png --output outputs/
```

### Use only a subset of models

```bash
python inference.py --input samples/ --paired --models baseline transfer attunet
```

### Also save each model's output as an individual PNG

```bash
python inference.py --input samples/ --paired --save-individual
```

### Custom models directory

```bash
python inference.py --input samples/ --paired --models-dir /path/to/models
```

### Full CLI reference

```bash
python inference.py --help
```

## Output

For each input image the script writes a single comparison PNG into the output directory:

```
outputs/comparison_<image_name>.png
```

The plot contains the following columns, in order:

1. **Input** (the tissue image after resizing to 256×256)

2. **Ground Truth** (only if `--paired` was used, so for the images of the MoNuSeg dataset)

3. One column per loaded model with that model's prediction

If `--save-individual` is set, each model's prediction is also written as a
separate PNG: `outputs/<image_name>_<model>.png`.


## MoNuSeg dataset

The models were trained on the publicly available **MoNuSeg** dataset
(MICCAI 2018 challenge): paired H&E-stained tissue images with manual
nuclei segmentation masks. The full dataset is available at
<https://monuseg.grand-challenge.org/>.

A small subset of paired sample images is included in `samples/` so the demo is self-contained.



## Authors

Azaliia & Jose Manuel — Master IA, Universidade de Vigo (2025-2026).
