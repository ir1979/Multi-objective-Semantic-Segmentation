text = """
def get_template_path(mode: str) -> str:
    from pathlib import Path
    return str(Path(__file__).parent.parent / "configs" / f"{mode}.yaml")

"""
with open("utils/config_wizard.py", "r") as f:
    orig = f.read()

with open("utils/config_wizard.py", "w") as f:
    f.write(orig.replace("def default_wizard_state", text + "def default_wizard_state"))

