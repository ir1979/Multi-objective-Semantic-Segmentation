import re
with open("utils/config_wizard.py", "r") as f:
    text = f.read()

# remove _set_nested function definition
text = re.sub(r'def _set_nested.*?^def default_wizard_state', 'def default_wizard_state', text, flags=re.MULTILINE|re.DOTALL)

with open("utils/config_wizard.py", "w") as f:
    f.write(text)

