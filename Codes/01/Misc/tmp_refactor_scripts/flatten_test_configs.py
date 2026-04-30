import re
import os

with open('tests/test_training.py', 'r') as f:
    text = f.read()

# Replace hierarchical self.base_config with flat properties
old_base_config = '''self.base_config = {
            "project": {"seed": 42},
            "data": {
                "rgb_dir": str(self.rgb_dir),
                "mask_dir": str(self.mask_dir),
                "image_size": self.image_size,
                "batch_size": 2,
                "train_ratio": 0.5,
                "val_ratio": 0.5,
                "test_ratio": 0.0,
            },
            "model": {
                "architecture": "unet",
                "encoder_filters": [16, 32, 64, 128],
                "dropout_rate": 0.1,
                "batch_norm": True,
                "deep_supervision": False,
            },
            "loss": {
                "strategy": "single",
                "pixel_type": "bce",
                "pixel_weight": 1.0,
                "boundary_enabled": False,
                "shape_enabled": False,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 1e-3,
                "lr_scheduler_type": "cosine",
                "warmup_epochs": 1,
                "min_learning_rate": 1e-5,
            },
        }'''

new_base_config = '''self.base_config = {
            "project_name": "test_project",
            "project_seed": 42,
            "data_rgb_dir": str(self.rgb_dir),
            "data_mask_dir": str(self.mask_dir),
            "data_image_size": self.image_size,
            "data_batch_size": 2,
            "data_train_ratio": 0.5,
            "data_val_ratio": 0.5,
            "data_test_ratio": 0.0,
            "model_architecture": "unet",
            "model_encoder_filters": [16, 32, 64, 128],
            "model_dropout_rate": 0.1,
            "model_batch_norm": True,
            "model_deep_supervision": False,
            "loss_strategy": "single",
            "loss_pixel_type": "bce",
            "loss_pixel_weight": 1.0,
            "loss_boundary_enabled": False,
            "loss_shape_enabled": False,
            "training_epochs": 1,
            "training_learning_rate": 1e-3,
            "training_lr_scheduler_type": "cosine",
            "training_warmup_epochs": 1,
            "training_min_learning_rate": 1e-5,
        }'''

text = text.replace(old_base_config, new_base_config)
with open('tests/test_training.py', 'w') as f:
    f.write(text)

