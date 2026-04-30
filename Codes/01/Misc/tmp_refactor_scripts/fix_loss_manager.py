import re
with open("losses/loss_manager.py", "r") as f:
    text = f.read()

repl = '''    def __post_init__(self) -> None:
        self.strategy = str(self.config.get("loss_strategy", "single")).lower()
        
        pixel_type = str(self.config.get("loss_pixel_type", "bce"))
        if pixel_type == "bce":
            self.pixel_loss = BCELoss()
        elif pixel_type == "iou":
            self.pixel_loss = IoULoss()
        elif pixel_type == "dice":
            self.pixel_loss = DiceLoss()
        elif pixel_type == "focal":
            self.pixel_loss = FocalLoss()
        else:
            raise ValueError(f"Unknown pixel loss type '{pixel_type}'.")

        self.boundary_enabled = bool(self.config.get("loss_boundary_enabled", False))
        self.shape_enabled = bool(self.config.get("loss_shape_enabled", False))
        self.boundary_loss = ApproxHausdorffLoss()
        self.shape_loss = ConvexityLoss()
        self.shape_reg_loss = RegularityLoss()
        self.weights = {
            "pixel": float(self.config.get("loss_pixel_weight", 1.0)),
            "boundary": float(self.config.get("loss_boundary_weight", 0.0)),
            "shape": float(self.config.get("loss_shape_weight", 0.0)),
        }
        self.deep_supervision_enabled = bool(self.config.get("model_deep_supervision", False))
        self.deep_supervision_weights = self.config.get("model_deep_supervision_weights", [0.5, 0.3, 0.2, 0.1])
'''
text = re.sub(r'    def __post_init__\(self\) -> None:\n[\s\S]*?(?=\n    def _apply_loss)', repl, text)
text = re.sub(r'def _build_pixel_loss.*?raise ValueError.*?$', '', text, flags=re.MULTILINE | re.DOTALL)

with open("losses/loss_manager.py", "w") as f:
    f.write(text)

