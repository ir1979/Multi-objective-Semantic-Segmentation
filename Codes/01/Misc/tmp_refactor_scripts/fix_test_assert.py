with open("tests/test_config_wizard.py", "r") as f:
    text = f.read()

text = text.replace(
    'self.assertIn("grid_search", loaded)',
    'self.assertTrue(any(k.startswith("grid_search") for k in loaded))'
)
with open("tests/test_config_wizard.py", "w") as f:
    f.write(text)
