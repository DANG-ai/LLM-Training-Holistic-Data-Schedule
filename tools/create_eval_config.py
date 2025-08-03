# Simple utility that takes an existing yml config file and adds evaluation specific parameters.

import argparse
import json
import os

def get_step_from_path(path):
    top_dir = path.split("/")[-1]
    if "global_step" in top_dir:
        return int(top_dir.split("global_step")[-1])
    else:
        return get_step_from_path("/".join(path.split("/")[:-1])) 

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file to modify.")
parser.add_argument("--num_fewshot", type=int, required=False, default=0, help="Flag fo the number of fewshot in-context examples to use. 0 if none.")
parser.add_argument("--iteration", type=int, required=False, default=None, help="Iteration of the model to evaluate. If not specified, will use the latest checkpoint.")

args = parser.parse_args()
model_step = get_step_from_path(args.config_path)

with open(args.config_path, "r") as f:
    config = json.load(f)

if "seed" not in config.keys():
    config["seed"] = 1234

config["wandb_run_name"] = f"seed{config['seed']}_eval"

eval_results_prefix = os.path.join(config["save"], f"step{model_step}")
config['load'] = config['save']
config['eval_results_prefix'] = eval_results_prefix

save_path = args.config_path.replace(".yml", "_eval.yml")

# if using iteration
if args.iteration is not None:
    config["iteration"] = args.iteration

# if using num_fewshot
if args.num_fewshot > 0:
    config["eval_num_fewshot"] = args.num_fewshot
    config["wandb_run_name"] += f"_{args.num_fewshot}shot"
    save_path = save_path.replace(".yml", f"_{args.num_fewshot}shot.yml")

with open(save_path, "w") as f:
    json.dump(config, f, indent=2)