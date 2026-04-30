with open("tests/test_config_wizard.py", "r") as f:
    text = f.read()

text = text.replace(
    'set(config["grid_search_parameters"].keys()),',
    'set([k.replace("grid_search_parameters_", "") for k in config if k.startswith("grid_search_parameters_")]),'
)
with open("tests/test_config_wizard.py", "w") as f:
    f.write(text)
