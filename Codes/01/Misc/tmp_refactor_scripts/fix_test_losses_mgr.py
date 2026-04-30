import re
with open("tests/test_losses.py", "r") as f:
    text = f.read()

repl_weighted = '{"loss_strategy": "weighted", "loss_pixel_type": "bce", "loss_pixel_weight": 1.0, "loss_boundary_enabled": True, "loss_boundary_weight": 0.5, "loss_shape_enabled": True, "loss_shape_weight": 0.25}'
text = re.sub(r'\{\s*"loss": \{\s*"strategy": "weighted".*?\}\s*\}', repl_weighted, text, flags=re.DOTALL)

repl_mgda = '{"loss_strategy": "single", "loss_pixel_type": "bce", "loss_pixel_weight": 1.0, "loss_boundary_enabled": True, "loss_boundary_weight": 0.2, "loss_shape_enabled": True, "loss_shape_weight": 0.1}'
text = re.sub(r'\{\s*"loss": \{\s*"strategy": "mgda".*?\}\s*\}', repl_mgda, text, flags=re.DOTALL)

with open("tests/test_losses.py", "w") as f:
    f.write(text)

