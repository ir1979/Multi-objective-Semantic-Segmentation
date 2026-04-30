import re
import os

def flatten(text):
    old = text
    # config["a"]["b"] done in previous script
    while True:
        # config.get("a", {})["b"] -> config.get("a_b")
        text = re.sub(r'config\.get\((["\'])([\w_]+)\1\s*,\s*\{\}\)\[(["\'])([\w_]+)\3\]', r'config.get("\2_\4")', text)
        if text == old: break
        old = text

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
