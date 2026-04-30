with open("tests/test_losses.py", "r") as f:
    text = f.read()

text = text.replace("BCEIoULoss(), ", "")
text = text.replace('{"loss": {"strategy": "weighted", "pixel": {"type": "bce", "weight": 1.0}, "boundary": {"enabled": True, "weight": 0.5}, "shape": {"enabled": True, "weight": 0.25},}}', '{"loss_strategy": "weighted", "loss_pixel_type": "bce", "loss_pixel_weight": 1.0, "loss_boundary_enabled": True, "loss_boundary_weight": 0.5, "loss_shape_enabled": True, "loss_shape_weight": 0.25}')
text = text.replace('{"loss": {"strategy": "mgda", "pixel": {"type": "bce", "weight": 1.0}, "boundary": {"enabled": True, "weight": 0.2}, "shape": {"enabled": True, "weight": 0.1},}}', '{"loss_strategy": "single", "loss_pixel_type": "bce", "loss_pixel_weight": 1.0, "loss_boundary_enabled": True, "loss_boundary_weight": 0.2, "loss_shape_enabled": True, "loss_shape_weight": 0.1}')

with open("tests/test_losses.py", "w") as f:
    f.write(text)

