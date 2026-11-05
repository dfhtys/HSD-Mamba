

import os
from numbers import Integral
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset


class ReconDataset(Dataset):


    SUPPORTED_VIEWS = (32, 64, 128, 256)

    def __init__(self, root, train=True, val=False, test=False, das=True, views=128):
        super().__init__()

        if val and test:
            raise ValueError("Choose either val=True or test=True, not both.")
        if not das:
            raise ValueError(
                "das=False is unsupported: ReconDataset requires paired "
                "sparse-view and full-view reconstruction images."
            )
        if (
            isinstance(views, bool)
            or not isinstance(views, Integral)
            or views not in self.SUPPORTED_VIEWS
        ):
            raise ValueError("views must be one of 32, 64, 128, or 256.")

        self.test_mode = bool(test)
        self.train_mode = bool(train) and not (val or test)
        self.val_mode = bool(val)
        self.views = int(views)
        self.input_key = "recon_img_{}".format(self.views)
        self.target_key = "recon_img_512"
        self.root = os.path.expanduser(os.fspath(root))
        if test:
            self.split = "Test"
        elif val:
            self.split = "Val"
        elif train:
            self.split = "Train"
        else:
            self.split = "Test"
        self.folder = Path(self.root) / self.split

        if not self.folder.is_dir():
            raise FileNotFoundError("Dataset directory not found: {}".format(self.folder))

        files = sorted(
            path for path in self.folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".mat"
        )
        if not files:
            raise FileNotFoundError("No .mat files found in: {}".format(self.folder))


        self.input_images = []
        self.target_images = []
        self.file_names = []
        self.image_shape = None

        for mat_path in files:
            try:
                matdata = scio.loadmat(
                    mat_path, variable_names=("SampleData", "ResultData")
                )
            except (OSError, ValueError, TypeError, NotImplementedError) as exc:
                raise ValueError(
                    "Cannot read MAT file {}: {}".format(mat_path, exc)
                ) from exc

            struct_data = self._get_structure(matdata, mat_path)
            input_image = self._read_image(struct_data, self.input_key, mat_path)
            target_image = self._read_image(struct_data, self.target_key, mat_path)

            if input_image.shape != target_image.shape:
                raise ValueError(
                    "{}: input shape {} does not match reference shape {}.".format(
                        mat_path, input_image.shape, target_image.shape
                    )
                )
            image_shape = input_image.shape[1:]
            if self.train_mode and image_shape[0] != image_shape[1]:
                raise ValueError(
                    "{}: training images must be square for shape-preserving "
                    "90-degree rotations; got {}.".format(mat_path, image_shape)
                )
            if self.image_shape is None:
                self.image_shape = image_shape
            elif image_shape != self.image_shape:
                raise ValueError(
                    "{}: image shape {} differs from {} in the same split. "
                    "This reader does not crop or resize images.".format(
                        mat_path, image_shape, self.image_shape
                    )
                )


            self.input_images.append(input_image)
            self.target_images.append(target_image)
            self.file_names.append(mat_path.name)

    @staticmethod
    def _get_structure(matdata, mat_path):
        for name in ("SampleData", "ResultData"):
            if name in matdata:
                array = np.asarray(matdata[name])
                if array.size != 1 or array.dtype.names is None:
                    raise ValueError(
                        "{}: {} must be a single MATLAB structure.".format(mat_path, name)
                    )
                return array.reshape(-1)[0]
        raise KeyError(
            "{}: neither SampleData nor ResultData was found.".format(mat_path)
        )

    @classmethod
    def _read_image(cls, struct_data, key, mat_path):
        if key not in struct_data.dtype.names:
            raise KeyError("{}: missing field '{}'.".format(mat_path, key))

        image = np.asarray(struct_data[key])
        if image.ndim != 2 or image.size == 0:
            raise ValueError(
                "{}: '{}' must be a nonempty 2-D image; got shape {}.".format(
                    mat_path, key, image.shape
                )
            )
        if image.dtype.kind not in "biuf":
            raise TypeError(
                "{}: '{}' must contain real numeric image values.".format(mat_path, key)
            )
        if not np.isfinite(image).all():
            raise ValueError("{}: '{}' contains NaN or infinity.".format(mat_path, key))

        with np.errstate(over="ignore", invalid="ignore"):
            image = np.ascontiguousarray(image[np.newaxis, :, :], dtype=np.float32)
        if not np.isfinite(image).all():
            raise ValueError(
                "{}: '{}' cannot be represented as finite float32 values.".format(mat_path, key)
            )
        return image

    @staticmethod
    def _normalize(image):
        minimum = image.min()
        value_range = image.max() - minimum
        if not torch.isfinite(value_range):
            raise ValueError("Image intensity range is too large for float32 normalization.")
        if value_range > 1e-6:
            return (image - minimum) / value_range

        return torch.zeros_like(image)

    def __getitem__(self, index):
        input_image = self._normalize(torch.from_numpy(self.input_images[index]))
        target_image = self._normalize(torch.from_numpy(self.target_images[index]))

        if self.train_mode:
            if torch.rand(1).item() < 0.5:
                input_image = torch.flip(input_image, dims=(2,))
                target_image = torch.flip(target_image, dims=(2,))
            if torch.rand(1).item() < 0.5:
                input_image = torch.flip(input_image, dims=(1,))
                target_image = torch.flip(target_image, dims=(1,))
            k = torch.randint(0, 4, (1,)).item()
            if k:
                input_image = torch.rot90(input_image, k, dims=(1, 2))
                target_image = torch.rot90(target_image, k, dims=(1, 2))

        input_image = input_image.contiguous()
        target_image = target_image.contiguous()
        return input_image, target_image

    def __len__(self):
        return len(self.input_images)
