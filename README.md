# HSD-Mamba

HSD-Mamba is an image-domain network for sparse-view photoacoustic image reconstruction. It maps an initial sparse-view reconstruction image to a refined reconstruction image using convolutional features, eight-direction Mamba modeling, adaptive directional fusion, and structure-enhancing skip connections.

## Files

- `hybrid_8way_mamba.py`: model implementation. The model entry point is `Hybrid_EightWayMamba`.
- `mydataset.py`: `ReconDataset`, a paired MAT-image reader for sparse-view inputs and full-view references.

This repository currently provides model and data-reading code only. Training scripts, evaluation scripts, pretrained weights, and processed datasets are not included.

## Dependencies

The source imports PyTorch (`torch`), `mamba-ssm` (`mamba_ssm`), NumPy, and SciPy. Other imports are from the Python standard library. Use a PyTorch/CUDA environment compatible with `mamba-ssm`; follow that package's installation instructions for your platform. No dependency versions are pinned here.

## Model usage

```python
import torch
from hybrid_8way_mamba import Hybrid_EightWayMamba

model = Hybrid_EightWayMamba(in_channels=1, channel=32).cuda().eval()
x = torch.randn(1, 1, 256, 256, device="cuda")
with torch.no_grad():
    prediction = model(x)  # [1, 1, 256, 256]
```

The supplied model uses `channel=32` by default, Mamba state dimension 16 and expansion factor 2. Its four skip paths use `StructureEnhancingSkip`. The final sigmoid produces image values in [0, 1].

## Dataset format

```text
dataset_root/
    Train/
        sample_001.mat
    Val/
        sample_002.mat
    Test/
        sample_003.mat
```

Each MAT file must be readable by `scipy.io.loadmat` and contain one scalar MATLAB structure named `SampleData` or `ResultData`. The selected structure must contain:

- `recon_img_32`, `recon_img_64`, `recon_img_128`, or `recon_img_256`: the selected sparse-view reconstruction, controlled by `views`.
- `recon_img_512`: the full-view reference image.

Only the selected sparse-view field and the reference field are required for a given reader instance. Both must be finite, nonempty, real-valued 2D arrays with matching dimensions. All samples in one split must have the same image size. HDF5-based MATLAB v7.3 files are not supported by this reader.

```python
from mydataset import ReconDataset

train_set = ReconDataset("/path/to/dataset_root", train=True, views=128)
val_set = ReconDataset("/path/to/dataset_root", train=False, val=True, views=128)
test_set = ReconDataset("/path/to/dataset_root", train=False, test=True, views=128)
input_image, reference_image = train_set[0]
# Both tensors have shape [1, H, W] and dtype float32.
```

The reader loads the selected split into memory. It does not crop or resize images. Input and reference images are independently min-max normalized per image; images with a range no greater than 1e-6 are returned as zeros. Training applies paired random horizontal/vertical flips and rotations by multiples of 90 degrees. Training images must be square. Validation and test samples receive no random augmentation.
