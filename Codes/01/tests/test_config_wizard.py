"""Tests for config wizard helper utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from utils.config_loader import ConfigValidator
from utils.config_wizard import (
    build_full_config,
    build_save_payload,
    default_wizard_state,
    estimate_grid_point_count,
    save_payload_to_file,
)


class TestConfigWizardHelpers(unittest.TestCase):
    """Validate config wizard serialization and search-space helpers."""

    def test_supported_architecture_passes_validation(self) -> None:
        state = default_wizard_state("single")
        state["model_architecture"] = "attunet"
        config = build_full_config(state)
        ConfigValidator().validate(config)

    def test_grid_search_count_uses_all_loss_dimensions(self) -> None:
        state = default_wizard_state("grid_search")
        state["grid_enabled"] = True
        state["grid_model_architecture_text"] = "unet, unetpp"
        state["grid_encoder_filters_text"] = "[64, 128, 256, 512, 1024]\n[32, 64, 128, 256, 512]"
        state["grid_pixel_loss_type_text"] = "bce, dice, bce"
        state["grid_boundary_loss_weight_text"] = "0.0, 0.3"
        state["grid_shape_loss_weight_text"] = "0.0, 0.1"
        state["grid_learning_rate_text"] = "1.0e-4, 5.0e-4"
        self.assertEqual(estimate_grid_point_count(state), 96)

    def test_grid_search_count_respects_include_flags(self) -> None:
        state = default_wizard_state("grid_search")
        state["grid_enabled"] = True
        state["grid_include_model_architecture"] = False
        state["grid_include_encoder_filters"] = False
        state["grid_include_pixel_loss_type"] = True
        state["grid_include_boundary_loss_weight"] = True
        state["grid_include_shape_loss_weight"] = False
        state["grid_include_learning_rate"] = True
        state["grid_pixel_loss_type_text"] = "bce, dice"
        state["grid_boundary_loss_weight_text"] = "0.0, 0.3, 0.5"
        state["grid_learning_rate_text"] = "1.0e-4, 5.0e-4"
        self.assertEqual(estimate_grid_point_count(state), 12)

    def test_build_full_config_only_emits_selected_grid_dimensions(self) -> None:
        state = default_wizard_state("grid_search")
        state["grid_enabled"] = True
        state["grid_include_model_architecture"] = False
        state["grid_include_encoder_filters"] = False
        state["grid_include_pixel_loss_type"] = True
        state["grid_include_boundary_loss_weight"] = False
        state["grid_include_shape_loss_weight"] = False
        state["grid_include_learning_rate"] = True
        state["grid_pixel_loss_type_text"] = "bce, dice"
        state["grid_learning_rate_text"] = "1.0e-4, 5.0e-4"

        config = build_full_config(state)

        self.assertEqual(
            set([k.replace("grid_search_parameters_", "") for k in config if k.startswith("grid_search_parameters_")]),
            {"pixel_loss_type", "learning_rate"},
        )

    def test_saved_payload_round_trips_as_yaml(self) -> None:
        state = default_wizard_state("grid_search")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "wizard.yaml"
            payload = build_save_payload(state, output_path)
            save_payload_to_file(payload, output_path)

            with output_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)

            self.assertIsInstance(loaded, dict)
            self.assertTrue(any(k.startswith("grid_search") for k in loaded))
