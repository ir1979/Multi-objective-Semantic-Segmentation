text = """
def config_to_wizard_state(resolved: dict, raw: dict, path: str) -> dict:
    state = {"mode": "single", "grid_enabled": False}
    state.update(resolved)
    grid_params = {k: v for k, v in resolved.items() if k.startswith("grid_search_parameters_")}
    if grid_params:
        state["grid_enabled"] = True
        state["mode"] = "grid_search"
        for k, v in grid_params.items():
            param = k.replace("grid_search_parameters_", "")
            state[f"grid_{param}_text"] = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
            state[f"grid_include_{param}"] = True
    return state
"""

with open("utils/config_wizard.py", "r") as f:
    orig = f.read()

with open("utils/config_wizard.py", "w") as f:
    f.write(orig.replace("def default_wizard_state", text + "def default_wizard_state"))

