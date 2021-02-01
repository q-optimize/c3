#!/usr/bin/python -u

import argparse
import hjson
from c3.utils.utils import num3str
from rich.console import Console
from rich.table import Table


parser = argparse.ArgumentParser()
parser.add_argument("log_file")
args = parser.parse_args()

log = None

try:
    with open(args.log_file) as file:
        log = hjson.load(file)
except FileNotFoundError:
    print("Logfile not found.")

if log:
    opt_map = log["opt_map"]
    optim_status = log["optim_status"]
    units = log["units"]
    params = optim_status["params"]
    grads = optim_status["gradient"]

    print(
        f"Optimization reached {optim_status['goal']:0.3g} at {optim_status['time']}\n"
    )
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter")
    table.add_column("Value", justify="right")
    table.add_column("Gradient", justify="right")
    for ii in range(len(opt_map)):
        equiv_ids = opt_map[ii]
        par = params[ii]
        grad = grads[ii]
        par = num3str(par)
        grad = num3str(grad)
        par_id = equiv_ids[0]
        nice_id = "-".join(par_id)
        table.add_row(nice_id, par + units[ii], grad + units[ii])
        if len(equiv_ids) > 1:
            for par_id in equiv_ids[1:]:
                nice_id = "-".join(par_id)
                table.add_row(nice_id, "''", "''")

console = Console()
console.print(table)
