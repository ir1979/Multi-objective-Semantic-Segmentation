import json
with open("/home/reza/.config/Code/User/workspaceStorage/f47c0959750656a5bfbaa59da2482a2f/GitHub.copilot-chat/transcripts/15ec33b7-ac26-4d61-97f4-9a0a4cac1fdc.jsonl") as f:
    for line in f:
        if "def config_to_wizard_state" in line:
            obj = json.loads(line)
            if "data" in obj and "content" in obj["data"]:
                print(obj["data"]["content"])
                break
