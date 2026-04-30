import re
def repl(f):
    with open(f, "r") as fd: text = fd.read()
    text = text.replace('        "bce_iou":           "#ff7f0e",\n', '')
    text = text.replace('bce_iou', 'bce')
    with open(f, "w") as fd: fd.write(text)

repl("visualization/style.py")
repl("utils/config_wizard.py")
repl("utils/config_validator.py")
