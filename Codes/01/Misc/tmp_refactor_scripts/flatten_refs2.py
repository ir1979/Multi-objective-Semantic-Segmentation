import re
import os

def flatten(text):
    # .get("a", {}).get("b")
    text = re.sub(r'\.get\((["\'])([\w_]+)\1\s*,\s*\{\}\)\.get\((["\'])([\w_]+)\3\)', r'.get("\2_\4")', text)
    # .get("a", {}).get("b", default)
    text = re.sub(r'\.get\((["\'])([\w_]+)\1\s*,\s*\{\}\)\.get\((["\'])([\w_]+)\3,\s*([^)]+)\)', r'.get("\2_\4", \5)', text)
    
    # .get("a").get("b") -> this is risky if .get("a") could be None, but assume dict
    text = re.sub(r'\.get\((["\'])([\w_]+)\1\)\.get\((["\'])([\w_]+)\3\)', r'.get("\2_\4")', text)
    text = re.sub(r'\.get\((["\'])([\w_]+)\1\)\.get\((["\'])([\w_]+)\3,\s*([^)]+)\)', r'.get("\2_\4", \5)', text)

    # ["a"]["b"] -> ["a_b"]
    while re.search(r'config\[(["\'])([\w_]+)\1\]\[(["\'])([\w_]+)\3\]', text):
        text = re.sub(r'config\[(["\'])([\w_]+)\1\]\[(["\'])([\w_]+)\3\]', r'config["\2_\4"]', text)

    return text

for r, d, f in os.walk('.'):
    for nm in f:
        if nm.endswith('.py') and not nm.startswith('flatten'):
            path = os.path.join(r, nm)
            with open(path, 'r') as fp:
                txt = fp.read()
            nxt = flatten(txt)
            if txt != nxt:
                with open(path, 'w') as fp:
                    fp.write(nxt)
                print("flattened", path)
