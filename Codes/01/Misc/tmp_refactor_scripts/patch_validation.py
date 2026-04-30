import re
with open("utils/config_loader.py", "r") as f:
    text = f.read()

# Replace required sections since we don't have them
text = re.sub(r'required_sections = \([\s\S]*?\n\s+\)', 'required_keys = ("project_name", "data_rgb_dir", "model_architecture", "loss_strategy")', text)
text = re.sub(r'for section in required_sections:\n\s+if section not in config:\n\s+raise ConfigValidationError\(f"Missing required config section: \'{section}\'"\)', 
              'for key in required_keys:\n            if key not in config:\n                raise ConfigValidationError(f"Missing required config key: \'{key}\'")', text)

text = re.sub(r'config\["model"\]\.get\("architecture", ""\)', r'config.get("model_architecture", "")', text)
text = re.sub(r'config\["loss"\]\.get\("strategy", ""\)', r'config.get("loss_strategy", "")', text)
text = re.sub(r'config\["loss"\]\.get\("pixel_type", ""\)', r'config.get("loss_pixel_type", "")', text)
text = re.sub(r'config\["training"\]\.get\("lr_scheduler_type", ""\)', r'config.get("training_lr_scheduler_type", "")', text)

text = re.sub(r'data = config\["data"\]\n\s+train_ratio = float\(data\.get\("train_ratio", 0\.0\)\)',
              r'train_ratio = float(config.get("data_train_ratio", 0.0))', text)
text = re.sub(r'val_ratio = float\(data\.get\("val_ratio", 0\.0\)\)',
              r'val_ratio = float(config.get("data_val_ratio", 0.0))', text)
text = re.sub(r'test_ratio = float\(data\.get\("test_ratio", 0\.0\)\)',
              r'test_ratio = float(config.get("data_test_ratio", 0.0))', text)

text = re.sub(r'data = config\["data"\]\n\s+for key in \("rgb_dir", "mask_dir"\):\n\s+value = data\.get\(key\)',
              r'for key in ("data_rgb_dir", "data_mask_dir"):\n            value = config.get(key)', text)

with open("utils/config_loader.py", "w") as f:
    f.write(text)

