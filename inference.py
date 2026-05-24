"""Pix2Pix Variants - MoNuSeg Inference Demo.

Loads several trained Pix2Pix generator variants and runs inference on one or
more input images. Produces side-by-side comparison plots and (optionally)
individual per-model output images.

Example:
    python inference.py --input samples/ --output outputs/ --paired
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# Model architectures 


class UNetBlock(nn.Module):
    """U-Net encoder block."""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2,
                              padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    """U-Net decoder block (concat handled externally)."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                          stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class Generator(nn.Module):
    """Baseline U-Net Generator for Pix2Pix."""
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super().__init__()
        # Encoder
        self.enc1 = UNetBlock(in_channels, ngf, normalize=False)
        self.enc2 = UNetBlock(ngf, ngf * 2)
        self.enc3 = UNetBlock(ngf * 2, ngf * 4)
        self.enc4 = UNetBlock(ngf * 4, ngf * 8)
        self.enc5 = UNetBlock(ngf * 8, ngf * 8)
        self.enc6 = UNetBlock(ngf * 8, ngf * 8)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        # Decoder
        self.dec6 = DecoderBlock(ngf * 8, ngf * 8, dropout=0.5)
        self.dec5 = DecoderBlock(ngf * 8 + ngf * 8, ngf * 8, dropout=0.5)
        self.dec4 = DecoderBlock(ngf * 8 + ngf * 8, ngf * 8, dropout=0.0)
        self.dec3 = DecoderBlock(ngf * 8 + ngf * 4, ngf * 4, dropout=0.0)
        self.dec2 = DecoderBlock(ngf * 4 + ngf * 2, ngf * 2, dropout=0.0)
        self.dec1 = DecoderBlock(ngf * 2 + ngf, ngf, dropout=0.0)
        # Final
        self.final_conv = nn.Conv2d(ngf, out_channels, kernel_size=3, padding=1)
        self.final = nn.Tanh()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        bottleneck = self.bottleneck(enc6)
        dec6 = self.dec6(bottleneck)
        dec6_cat = torch.cat([dec6, enc5], dim=1)
        dec5 = self.dec5(dec6_cat)
        dec5_cat = torch.cat([dec5, enc4], dim=1)
        dec4 = self.dec4(dec5_cat)
        dec4_cat = torch.cat([dec4, enc3], dim=1)
        dec3 = self.dec3(dec4_cat)
        dec3_cat = torch.cat([dec3, enc2], dim=1)
        dec2 = self.dec2(dec3_cat)
        dec2_cat = torch.cat([dec2, enc1], dim=1)
        dec1 = self.dec1(dec2_cat)
        return self.final(self.final_conv(dec1))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet-based generator (CycleGAN-style)."""
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_residual_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        in_features = ngf
        for _ in range(2):
            out_features = in_features * 2
            layers += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        for _ in range(n_residual_blocks):
            layers += [ResidualBlock(in_features)]
        for _ in range(2):
            out_features = in_features // 2
            layers += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNetGenerator(nn.Module):
    """U-Net Generator with Attention Gates on skip connections."""
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, ngf, normalize=False)
        self.enc2 = UNetBlock(ngf, ngf * 2)
        self.enc3 = UNetBlock(ngf * 2, ngf * 4)
        self.enc4 = UNetBlock(ngf * 4, ngf * 8)
        self.enc5 = UNetBlock(ngf * 8, ngf * 8)
        self.enc6 = UNetBlock(ngf * 8, ngf * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        self.dec6 = DecoderBlock(ngf * 8, ngf * 8, dropout=0.5)
        self.dec5 = DecoderBlock(ngf * 8 + ngf * 8, ngf * 8, dropout=0.5)
        self.dec4 = DecoderBlock(ngf * 8 + ngf * 8, ngf * 8, dropout=0.0)
        self.dec3 = DecoderBlock(ngf * 8 + ngf * 4, ngf * 4, dropout=0.0)
        self.dec2 = DecoderBlock(ngf * 4 + ngf * 2, ngf * 2, dropout=0.0)
        self.dec1 = DecoderBlock(ngf * 2 + ngf, ngf, dropout=0.0)
        self.att5 = AttentionGate(g_channels=ngf * 8, x_channels=ngf * 8, inter_channels=ngf * 4)
        self.att4 = AttentionGate(g_channels=ngf * 8, x_channels=ngf * 8, inter_channels=ngf * 4)
        self.att3 = AttentionGate(g_channels=ngf * 8, x_channels=ngf * 4, inter_channels=ngf * 2)
        self.att2 = AttentionGate(g_channels=ngf * 4, x_channels=ngf * 2, inter_channels=ngf)
        self.att1 = AttentionGate(g_channels=ngf * 2, x_channels=ngf,     inter_channels=ngf // 2)
        self.final_conv = nn.Conv2d(ngf, out_channels, kernel_size=3, padding=1)
        self.final = nn.Tanh()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        bottleneck = self.bottleneck(enc6)
        dec6 = self.dec6(bottleneck)
        enc5_att = self.att5(g=dec6, x=enc5)
        dec6_cat = torch.cat([dec6, enc5_att], dim=1)
        dec5 = self.dec5(dec6_cat)
        enc4_att = self.att4(g=dec5, x=enc4)
        dec5_cat = torch.cat([dec5, enc4_att], dim=1)
        dec4 = self.dec4(dec5_cat)
        enc3_att = self.att3(g=dec4, x=enc3)
        dec4_cat = torch.cat([dec4, enc3_att], dim=1)
        dec3 = self.dec3(dec4_cat)
        enc2_att = self.att2(g=dec3, x=enc2)
        dec3_cat = torch.cat([dec3, enc2_att], dim=1)
        dec2 = self.dec2(dec3_cat)
        enc1_att = self.att1(g=dec2, x=enc1)
        dec2_cat = torch.cat([dec2, enc1_att], dim=1)
        dec1 = self.dec1(dec2_cat)
        return self.final(self.final_conv(dec1))


def _build_transfer_generator():
    """Build the transfer-learning generator. Requires segmentation_models_pytorch."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        return None

    class TransferLearningGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,  # weights come from the .pth checkpoint
                in_channels=3,
                classes=3,
                activation=None,
            )
            self.tanh = nn.Tanh()

        def forward(self, x):
            return self.tanh(self.unet(x))

    return TransferLearningGenerator()



# Registering the models: name -> {filename, builder}


MODELS_CONFIG = {
    'baseline':  {'filename': 'best_generator.pth',                   'builder': lambda: Generator()},
    'lsgan':     {'filename': 'best_generator_lsgan_l1.pth',          'builder': lambda: Generator()},
    'bce_l2':    {'filename': 'best_generator_bce_l2.pth',            'builder': lambda: Generator()},
    'l1_only':   {'filename': 'best_generator_l1_only.pth',           'builder': lambda: Generator()},
    'augmented': {'filename': 'best_generator_augmented.pth',         'builder': lambda: Generator()},
    'resnet':    {'filename': 'best_generator_resnet.pth',            'builder': lambda: ResNetGenerator(n_residual_blocks=9)},
    'attunet':   {'filename': 'best_generator_attunet.pth',           'builder': lambda: AttentionUNetGenerator()},
    'transfer':  {'filename': 'best_generator_transfer_resnet34.pth', 'builder': _build_transfer_generator},
}




def load_image_pair(path, image_size, paired):
    """Load an image. If paired=True, split (left=input, right=GT)."""
    img = Image.open(path).convert('RGB')
    if paired:
        w, h = img.size
        input_img = img.crop((0, 0, w // 2, h)).resize((image_size, image_size), Image.BILINEAR)
        gt_img    = img.crop((w // 2, 0, w, h)).resize((image_size, image_size), Image.BILINEAR)
        return input_img, gt_img
    return img.resize((image_size, image_size), Image.BILINEAR), None


def pil_to_tensor(pil_img):
    """PIL [0,255] -> tensor [-1,1] of shape (1,3,H,W)."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def tensor_to_pil(tensor):
    """Tensor [-1,1] of shape (1,3,H,W) -> PIL image."""
    arr = tensor.squeeze(0).detach().cpu().numpy()
    arr = np.clip((arr + 1) / 2, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8).transpose(1, 2, 0))



# Main


def main():
    parser = argparse.ArgumentParser(
        description="Pix2Pix variants - MoNuSeg inference demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', required=True,
                        help="Input image path or directory of images.")
    parser.add_argument('--output', default='./outputs',
                        help="Directory for output comparison plots.")
    parser.add_argument('--models-dir', default='./models',
                        help="Directory containing the trained .pth files.")
    parser.add_argument('--models', nargs='+', default=list(MODELS_CONFIG.keys()),
                        choices=list(MODELS_CONFIG.keys()),
                        help="Subset of models to run.")
    parser.add_argument('--paired', action='store_true',
                        help="Treat input as paired (left=input, right=GT) and split.")
    parser.add_argument('--image-size', type=int, default=256,
                        help="Square size to resize input(s) to.")
    parser.add_argument('--device', default=None,
                        help="cpu / cuda (auto-detect if not set).")
    parser.add_argument('--save-individual', action='store_true',
                        help="Also save each model's output as a separate PNG.")
    args = parser.parse_args()

    # Device
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[info] Using device: {device}")

    os.makedirs(args.output, exist_ok=True)

    # Collect input image paths
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
    else:
        raise FileNotFoundError(f"Input path does not exist: {args.input}")

    if not image_paths:
        raise ValueError(f"No images found at: {args.input}")
    print(f"[info] Found {len(image_paths)} image(s) to process.")

    # Load all requested models that have a .pth available
    print(f"[info] Loading models from: {args.models_dir}")
    models = {}
    for name in args.models:
        cfg = MODELS_CONFIG[name]
        model_path = Path(args.models_dir) / cfg['filename']
        if not model_path.exists():
            print(f"  [skip] {name}: {cfg['filename']} not found")
            continue
        model = cfg['builder']()
        if model is None:
            print(f"  [skip] {name}: builder returned None (missing dependency, "
                  f"e.g. 'segmentation-models-pytorch' for 'transfer').")
            continue
        try:
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.to(device).eval()
            models[name] = model
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  [ok]   {name:<12} ({cfg['filename']}, {n_params:,} params)")
        except Exception as e:
            print(f"  [skip] {name}: load failed ({type(e).__name__}: {e})")

    if not models:
        raise RuntimeError("No models loaded. Check --models-dir and that .pth files are there.")

    # Process each image
    for img_path in image_paths:
        print(f"\n[run]  {img_path.name}")
        input_pil, gt_pil = load_image_pair(img_path, args.image_size, args.paired)
        input_tensor = pil_to_tensor(input_pil).to(device)

        outputs = {}
        with torch.no_grad():
            for name, model in models.items():
                outputs[name] = tensor_to_pil(model(input_tensor))

        # Comparison plot
        n_cols = 1 + (1 if gt_pil is not None else 0) + len(outputs)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))
        if n_cols == 1:
            axes = [axes]

        col = 0
        axes[col].imshow(input_pil); axes[col].set_title("Input", fontsize=10); axes[col].axis('off'); col += 1
        if gt_pil is not None:
            axes[col].imshow(gt_pil); axes[col].set_title("Ground Truth", fontsize=10); axes[col].axis('off'); col += 1
        for name, out_pil in outputs.items():
            axes[col].imshow(out_pil); axes[col].set_title(name, fontsize=10); axes[col].axis('off'); col += 1

        plt.tight_layout()
        out_path = Path(args.output) / f"comparison_{img_path.stem}.png"
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [save] {out_path}")

        if args.save_individual:
            for name, out_pil in outputs.items():
                p = Path(args.output) / f"{img_path.stem}_{name}.png"
                out_pil.save(p)
                print(f"  [save] {p}")

    print(f"\n[done] All results in: {args.output}")


if __name__ == '__main__':
    main()
