import re
with open("utils/config_loader.py", "r") as f:
    text = f.read()

def repl(text):
    text = re.sub(r'def load_config.*? loaded', 
                  'def load_config(path: str) -> Dict[str, Any]:\n    config_path = Path(path)\n    if not config_path.exists():\n        raise ConfigValidationError(f"Configuration file not found: {config_path}")\n\n    loaded = _load_yaml(config_path)\n    loaded = config_to_flat_dict(loaded)', text, flags=re.DOTALL)
    
    # Also change config_to_flat_dict separator to be `_` instead of `.` to match python dictionary keys!
    text = re.sub(r'full_key = f"\{prefix\}\.\{key\}"', r'full_key = f"{prefix}_{key}"', text)
    return text

with open("utils/config_loader.py", "w") as f:
    f.write(repl(text))

