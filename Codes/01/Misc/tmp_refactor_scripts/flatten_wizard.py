import re
with open("utils/config_wizard.py", "r") as f:
    text = f.read()

# Replace _set_nested(config, ("a", "b"), val) with config['a_b'] = val
text = re.sub(
    r'_set_nested\(\s*config,\s*\(\s*(["\'][^"\']+["\'](?:\s*,\s*["\'][^"\']+["\'])*)\s*\),\s*(.+?)\s*\)',
    lambda m: f"config[{'_'.join(tuple(eval('[' + m.group(1) + ']')))!r}] = {m.group(2)}",
    text
)

with open("utils/config_wizard.py", "w") as f:
    f.write(text)

