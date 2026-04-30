with open("losses/__init__.py", "r") as f:
    text = f.read()

text = text.replace("BCEIoULoss, ", "")

with open("losses/__init__.py", "w") as f:
    f.write(text)

with open("tests/test_losses.py", "r") as f:
    text = f.read()

text = text.replace("BCEIoULoss, ", "")
text = text.replace("from losses.pixel_losses import BCEIoULoss\n", "")

# Remove test_bce_iou_loss since the class is deleted
import re
text = re.sub(r'def test_bce_iou_loss\(self\) -> None:\n[\s\S]*?(?=^    def test_)', '', text, flags=re.MULTILINE)

with open("tests/test_losses.py", "w") as f:
    f.write(text)

