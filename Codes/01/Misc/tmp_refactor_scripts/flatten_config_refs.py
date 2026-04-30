import re
import os

def flatten_refs(content):
    # Match config.get("section", {}).get("key", default)
    content = re.sub(
        r'config\.get\((["\'])([^"\']+)\1,\s*\{\}\)\.get\((["\'])([^"\']+)\3(?:,\s*([^)]+))?\)',
        lambda m: f'config.get("{m.group(2)}_{m.group(4)}"{", " + m.group(5) if m.group(5) else ""})',
        content
    )
    # Match config["section"]["key"]
    content = re.sub(
        r'config\[(["\'])([^"\']+)\]\[(["\'])([^"\']+)\]',
        lambda m: f'config["{m.group(2)}_{m.group(4)}"]',
        content
    )
    # Match config.get("section", {})["key"]
    content = re.sub(
        r'config\.get\((["\'])([^"\']+)\](?:,\s*\{\})\)\[(["\'])([^"\']+)\]',
        lambda m: f'config["{m.group(2)}_{m.group(4)}"]',
        content
    )
    return content

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py') and file != 'flatten_config_refs.py':
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            new_content = flatten_refs(content)
            if new_content != content:
                with open(path, 'w') as f:
                    f.write(new_content)
                print(f"Updated {path}")
