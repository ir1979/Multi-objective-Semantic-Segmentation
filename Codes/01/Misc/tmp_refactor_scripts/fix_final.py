import re
with open("tests/test_run_all_cli.py", "r") as f:
    text = f.read()
text = text.replace('updated["logging"]["console_level"]', 'updated["logging_console_level"]')
text = text.replace('updated["logging"]["file_level"]', 'updated["logging_file_level"]')
with open("tests/test_run_all_cli.py", "w") as f:
    f.write(text)

