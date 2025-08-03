# Simple utility to create a yml config file from command line arguments.
# Handles int, float, bool, and string arguments.

import sys
import os
import json

RUN_SPECIFIC_CONFIG_PATH="hds_config/run_specific"
os.makedirs(RUN_SPECIFIC_CONFIG_PATH, exist_ok=True)

print(sys.argv[1:])
config = {}
for k, v in zip(sys.argv[1::2], sys.argv[2::2]):
    k = k.replace("--", "")
    # first, handle numeric inputs
    try:
        f_v = float(v)
        if f_v.is_integer():
            v = int(v)
        else:
            v = float(v)
    # if not numeric, then convert bools
    except:
        if v.lower() == "true":
            v = True
        elif v.lower() == "false":
            v = False
    # otherwise, it's a string and we do nothing
    config[k] = v

print(config)

assert("save" in config)
save_path = os.path.join(RUN_SPECIFIC_CONFIG_PATH, config["save"].split("/")[-1] + ".yml")
with open(save_path, "w") as f:
    json.dump(config, f, indent=2)

