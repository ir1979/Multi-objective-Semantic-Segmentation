import re
with open("Misc/run_all.py", "r") as f:
    text = f.read()

text = re.sub(r'config\.setdefault\("logging", \{\}\)\["console_level"\]\s*=\s*(.+)', r'config["logging_console_level"] = \1', text)
text = re.sub(r'config\.setdefault\("logging", \{\}\)\["file_level"\]\s*=\s*(.+)', r'config["logging_file_level"] = \1', text)

with open("Misc/run_all.py", "w") as f:
    f.write(text)

