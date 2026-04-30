with open("tests/test_run_all_cli.py", "r") as f:
    text = f.read()

text = text.replace('{"logging": {"file_level": "DEBUG", "validation_image_interval": 5}}', '{"logging_file_level": "DEBUG", "logging_validation_image_interval": 5}')
text = text.replace('updated["logging"]["console_level"]', 'updated["logging_console_level"]')
text = text.replace('updated["logging"]["file_level"]', 'updated["logging_file_level"]')
text = text.replace('updated["logging"]["validation_image_interval"]', 'updated["logging_validation_image_interval"]')
text = text.replace('config["logging"]["file_level"] = "INFO"', 'config["logging_file_level"] = "INFO"')

with open("tests/test_run_all_cli.py", "w") as f:
    f.write(text)

with open("Misc/run_all.py", "r") as f:
    ra = f.read()
ra = ra.replace('config.setdefault("logging", {})["console_level"]', 'config["logging_console_level"]')
ra = ra.replace('config["logging"]["file_level"] =', 'config["logging_file_level"] =')    
with open("Misc/run_all.py", "w") as f:
    f.write(ra)
