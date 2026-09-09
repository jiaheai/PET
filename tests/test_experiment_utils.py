from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiment_utils import (
    experiment_metadata,
    normalize_split,
    prepare_results_file,
)
from nifti_loader import PatientVolumes
import c_2_sweep


def make_patient(
    patient_id: str,
    cohort: str,
    value: float,
    label: int,
) -> PatientVolumes:
    pet = np.full((2, 2, 2), value, dtype=np.float32)
    mask = np.ones_like(pet, dtype=bool)
    return PatientVolumes(
        patient_id,
        cohort,
        pet,
        mask,
        pet.copy(),
        np.eye(4),
        label,
    )


class NormalizationTests(unittest.TestCase):
    def test_heldout_values_do_not_affect_fitted_transform(self) -> None:
        target_fit = [
            make_patient("a1", "AUGSBURG", 1, 0),
            make_patient("a2", "AUGSBURG", 3, 1),
        ]
        source_train = [
            make_patient("s1", "SWISS", 10, 0),
            make_patient("s2", "SWISS", 14, 1),
        ]

        def normalized_with_heldout(value: float) -> dict:
            heldout = make_patient("a3", "AUGSBURG", value, 0)
            cohorts = {
                "AUGSBURG": [*target_fit, heldout],
                "SWISS": source_train,
            }
            return normalize_split(
                all_cohorts=cohorts,
                cohort_train={"AUGSBURG": [], "SWISS": source_train},
                cohort_val={"AUGSBURG": [], "SWISS": []},
                target_cohort="AUGSBURG",
                target_harmonization=target_fit,
                target_heldout=[heldout],
            )[0]

        first = normalized_with_heldout(100)
        second = normalized_with_heldout(100_000)
        np.testing.assert_allclose(
            first["AUGSBURG"][0].pet,
            second["AUGSBURG"][0].pet,
        )
        np.testing.assert_allclose(
            first["SWISS"][0].pet,
            second["SWISS"][0].pet,
        )


class ProvenanceTests(unittest.TestCase):
    def test_fingerprint_changes_with_data_and_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cohort = root / "AUGSBURG"
            cohort.mkdir()
            data_file = cohort / "1_PET_res_0.nii.gz"
            data_file.write_bytes(b"first")
            common = {
                "model": "test",
                "data_path": root,
                "cohort_names": ["AUGSBURG"],
                "zscore_correction": False,
                "code_paths": [Path(__file__)],
            }
            first = experiment_metadata(parameters={"x": 1}, **common)
            changed_parameter = experiment_metadata(
                parameters={"x": 2},
                **common,
            )
            self.assertNotEqual(
                first["experiment_id"],
                changed_parameter["experiment_id"],
            )

            data_file.write_bytes(b"second")
            changed_data = experiment_metadata(parameters={"x": 1}, **common)
            self.assertNotEqual(
                first["experiment_id"],
                changed_data["experiment_id"],
            )

    def test_stale_and_duplicate_runs_are_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            good = {
                "record_type": "run",
                "experiment_id": "current",
                "torch_seed": 0,
                "target_cohort": "AUGSBURG",
                "target_heldout": {"auc": 0.5},
            }
            stale = {**good, "experiment_id": "old"}
            path.write_text(json.dumps(good) + "\n" + json.dumps(stale) + "\n")

            kept, pending = prepare_results_file(
                path=path,
                fresh=False,
                experiment_id="current",
                cohort_names=["AUGSBURG"],
                torch_seeds=[0],
            )
            self.assertEqual(len(kept), 1)
            self.assertEqual(pending, [])

            with path.open("a") as handle:
                handle.write(json.dumps(good) + "\n")
            kept, pending = prepare_results_file(
                path=path,
                fresh=False,
                experiment_id="current",
                cohort_names=["AUGSBURG"],
                torch_seeds=[0],
            )
            self.assertEqual(kept, [])
            self.assertEqual(pending, [0])


class ClassifierSplitTests(unittest.TestCase):
    def test_c_classifier_fits_only_the_supplied_training_subset(self) -> None:
        source = [
            make_patient(f"s{index}", "SWISS", index, index % 2)
            for index in range(10)
        ]
        target = [
            make_patient(f"a{index}", "AUGSBURG", index, index % 2)
            for index in range(4)
        ]
        classifier_train = source[:6]
        fit_sizes: list[int] = []

        class FakeClassifier:
            def fit(self, features, labels):
                fit_sizes.append(len(features))
                return self

            def predict_proba(self, features):
                probability = np.full(len(features), 0.5)
                return np.column_stack([1 - probability, probability])

        def fake_encode(encoders, order, patients):
            features = np.zeros((len(patients), 2))
            labels = np.asarray([patient.label for patient in patients])
            return features, labels

        with (
            patch.object(c_2_sweep, "encode_concat", side_effect=fake_encode),
            patch.object(
                c_2_sweep,
                "make_pipeline",
                return_value=FakeClassifier(),
            ),
        ):
            c_2_sweep.evaluate_target(
                0,
                {},
                {"AUGSBURG": target, "SWISS": source},
                "AUGSBURG",
                ["SWISS"],
                classifier_train,
                target[:2],
                target[2:],
            )

        self.assertEqual(fit_sizes, [len(classifier_train)])


if __name__ == "__main__":
    unittest.main()
