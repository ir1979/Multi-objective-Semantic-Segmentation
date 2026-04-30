import re
with open("utils/config_loader.py", "r") as f:
    text = f.read()

text = text.replace("loaded = config_to_flat_dict(loaded) = _load_yaml(config_path)", "loaded = config_to_flat_dict(loaded)")

with open("utils/config_loader.py", "w") as f:
    f.write(text)

