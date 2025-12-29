#!/usr/bin/env python3
"""
Script to export a trained SCNN model to TorchScript format.

Uses torch.jit.trace with fixed input size. Default 288x952 preserves
KITTI aspect ratio (370x1226 -> 288x952, divisible by 8).
"""

import argparse
import os

from scnn_torch.model import SCNN

import torch
from torch.nn import Module


class SCNNWrapper(Module):
    """Wrapper for SCNN model for TorchScript export."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        print('Loading SCNN model from checkpoint...')
        self.model = SCNN(ms_ks=9, pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()

        print(f'Loaded from iteration {checkpoint.get("iteration", "unknown")}')

    def forward(self, x):
        seg_pred, exist_pred = self.model(x)
        return seg_pred, exist_pred


def export_scnn_model(checkpoint_path, output_path, input_height, input_width):
    """Export SCNN model to TorchScript format using tracing."""
    print('Creating SCNN model wrapper...')
    model = SCNNWrapper(checkpoint_path)

    print('Preparing dummy input...')
    dummy_input = torch.randn(1, 3, input_height, input_width)

    print('Exporting to TorchScript using trace...')
    try:
        traced_module = torch.jit.trace(model, dummy_input, strict=False)
        traced_module.save(output_path)
        print(f'TorchScript model saved to: {output_path}')
    except Exception as e:
        print(f'TorchScript export failed: {e}')
        raise

    # Test the exported model
    print(f'\nTesting TorchScript model with input size {input_height}x{input_width}...')
    try:
        loaded_model = torch.jit.load(output_path)
        seg_output, exist_output = loaded_model(dummy_input)
        print('TorchScript model validation passed')
        print(f'Segmentation output shape: {seg_output.shape}')
        print(f'Existence output shape: {exist_output.shape}')
    except Exception as e:
        print(f'TorchScript model validation failed: {e}')


if __name__ == '__main__':
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=288,
                    help='The height of the input image (default: 288)')
    ap.add_argument('--width', type=int, default=952,
                    help='The width of the input image (default: 952 for KITTI aspect ratio)')
    ap.add_argument('--checkpoint', type=str, required=True,
                    help='Path to checkpoint file')
    ap.add_argument('--output-dir', type=str, default='models',
                    help='The path to output .pt file')
    args = vars(ap.parse_args())

    # Create output directory if it doesn't exist
    os.makedirs(args['output_dir'], exist_ok=True)

    height = args['height']
    width = args['width']
    checkpoint = args['checkpoint']
    output_dir = args['output_dir']

    # Export to TorchScript
    print(f'=== Exporting SCNN for input size: {height}x{width} ===')
    output_path = os.path.join(output_dir, f'scnn_vgg16_{height}x{width}.pt')
    export_scnn_model(
        checkpoint_path=checkpoint,
        output_path=output_path,
        input_height=height,
        input_width=width
    )

    print('TorchScript export completed.')
