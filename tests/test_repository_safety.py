from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import a_2
import a_3
import b_2
import b_3
import c_2
import c_3
import cnn_2
import cnn_3
import visualizer
from experiment_utils import seed_everything
from nifti_loader import load_cohort, load_patient


class NiftiValidationTests(unittest.TestCase):
    def _save_pair(
        self,
        root: Path,
        *,
        label: int = 0,
        pet: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        pet_affine: np.ndarray | None = None,
        mask_affine: np.ndarray | None = None,
    ) -> tuple[Path, Path]:
        pet = np.ones((4, 4, 4), dtype=np.float32) if pet is None else pet
        mask = np.ones((4, 4, 4), dtype=np.float32) if mask is None else mask
        pet_affine = np.eye(4) if pet_affine is None else pet_affine
        mask_affine = np.eye(4) if mask_affine is None else mask_affine
        pet_path = root / f"1_PET_res_{label}.nii.gz"
        mask_path = root / "1_prostate_mask_res.nii.gz"
        nib.save(nib.Nifti1Image(pet, pet_affine), pet_path)
        nib.save(nib.Nifti1Image(mask, mask_affine), mask_path)
        return pet_path, mask_path

    def test_rejects_invalid_pet_mask_pairs(self) -> None:
        cases = {
            "shape": {
                "mask": np.ones((3, 4, 4), dtype=np.float32),
                "match": "shape mismatch",
            },
            "affine": {
                "mask_affine": np.diag([2.0, 1.0, 1.0, 1.0]),
                "match": "affine mismatch",
            },
            "nonfinite_pet": {
                "pet": np.full((4, 4, 4), np.nan, dtype=np.float32),
                "match": "non-finite",
            },
            "empty_mask": {
                "mask": np.zeros((4, 4, 4), dtype=np.float32),
                "match": "empty",
            },
            "label": {"label": 2, "match": "binary risk label"},
        }
        for name, options in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                match = options.pop("match")
                pet_path, mask_path = self._save_pair(Path(temp), **options)
                with self.assertRaisesRegex(ValueError, match):
                    load_patient(pet_path, mask_path)

    def test_rejects_duplicate_patient_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._save_pair(root, label=0)
            nib.save(
                nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)),
                root / "1_PET_res_1.nii.gz",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate patient ID"):
                load_cohort(root)


class RepositoryConfigurationTests(unittest.TestCase):
    def test_standalone_dataset_defaults_exist(self) -> None:
        modules = [a_2, a_3, b_2, b_3, c_2, c_3, cnn_2, cnn_3]
        for module in modules:
            with self.subTest(module=module.__name__):
                data_path = Path(module.__file__).parent / module.DATA_PATH
                self.assertTrue(data_path.is_dir(), data_path)
        self.assertEqual(a_2.COHORT_NAMES, ["AUGSBURG", "SWISS"])
        self.assertTrue(visualizer.DATA_ROOT.is_dir())

    def test_all_autoencoder_decoders_have_linear_outputs(self) -> None:
        decoder_classes = [
            a_2.Decoder3D,
            a_3.Decoder3D,
            b_2.Decoder3D,
            b_3.Decoder3D,
            c_2.Decoder3D,
            c_3.Decoder3D,
        ]
        for decoder_class in decoder_classes:
            with self.subTest(decoder=decoder_class.__module__):
                decoder = decoder_class()
                self.assertIsInstance(decoder.deconv[-1], torch.nn.Identity)

    def test_seed_everything_reproduces_all_rng_streams(self) -> None:
        seed_everything(73)
        first = (random.random(), np.random.random(), torch.rand(4))
        seed_everything(73)
        second = (random.random(), np.random.random(), torch.rand(4))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_shared_batchnorm_is_cohort_order_neutral(self) -> None:
        cohort_a = torch.zeros(2, 1, 32, 32, 32)
        cohort_b = torch.full((3, 1, 32, 32, 32), 4.0)

        for model_class in [a_2.HarmonizationModel, a_3.HarmonizationModel]:
            with self.subTest(model=model_class.__module__):
                seed_everything(19)
                forward_order = model_class()
                reverse_order = copy.deepcopy(forward_order)
                forward_order.train()
                reverse_order.train()

                encoder_batch_sizes = []
                hook = forward_order.encoder.register_forward_pre_hook(
                    lambda _module, args: encoder_batch_sizes.append(args[0].size(0))
                )
                with torch.no_grad():
                    x_hats, zs = forward_order.reconstruct_cohorts(
                        {"A": cohort_a, "B": cohort_b}
                    )
                    reverse_order.reconstruct_cohorts(
                        {"B": cohort_b, "A": cohort_a}
                    )
                hook.remove()

                self.assertEqual(encoder_batch_sizes, [5])
                self.assertEqual(x_hats["A"].size(0), 2)
                self.assertEqual(zs["B"].size(0), 3)

                forward_bn = [
                    module
                    for module in forward_order.encoder.modules()
                    if isinstance(module, torch.nn.BatchNorm3d)
                ]
                reverse_bn = [
                    module
                    for module in reverse_order.encoder.modules()
                    if isinstance(module, torch.nn.BatchNorm3d)
                ]
                for first, second in zip(forward_bn, reverse_bn):
                    torch.testing.assert_close(
                        first.running_mean,
                        second.running_mean,
                        rtol=1e-6,
                        atol=1e-7,
                    )
                    torch.testing.assert_close(
                        first.running_var,
                        second.running_var,
                        rtol=1e-6,
                        atol=1e-7,
                    )


if __name__ == "__main__":
    unittest.main()
