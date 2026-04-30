import re
with open("Misc/run_all.py", "r") as f:
    text = f.read()

repl = '''def apply_logging_overrides(config: Dict[str, object], console_level: str, file_level: str) -> Dict[str, object]:
    """Apply CLI logging overrides while keeping file logs verbose."""
    resolved = dict(config)
    resolved["logging_console_level"] = normalize_log_level(console_level)
    resolved["logging_file_level"] = normalize_log_level(file_level)
    return resolved'''

text = re.sub(r'def apply_logging_overrides\(.*?\)\s*->\s*Dict\[str, object\]:[\s\S]*?return resolved', repl, text)

with open("Misc/run_all.py", "w") as f:
    f.write(text)

