#!/usr/bin/env python3

import time
import argparse
import hjson
from typing import Any, Dict

from c3.utils.utils import num3str
from rich.console import Console
from rich.table import Table


def show_table(log: Dict[str, Any], console: Console) -> None:
    """Generate a rich table from an optimization status and display it on the console.

    Parameters
    ----------
    log : Dict
        Dictionary read from a json log file containing a c3-toolset optimization status.
    console : Console
        Rich console for output.
    """
    if log:
        opt_map = log["opt_map"]
        optim_status = log["optim_status"]
        units = log["units"]
        params = optim_status["params"]
        grads = optim_status["gradient"]

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Parameter")
        table.add_column("Value", justify="right")
        table.add_column("Gradient", justify="right")
        for ii, equiv_ids in enumerate(opt_map):
            par = params[ii]
            grad = grads[ii]
            par = num3str(par)
            grad = num3str(grad)
            par_id = equiv_ids[0]
            table.add_row(par_id, par + units[ii], grad + units[ii])
            if len(equiv_ids) > 1:
                for par_id in equiv_ids[1:]:
                    table.add_row(par_id, "''", "''")

        console.clear()
        print(
            f"Optimization reached {optim_status['goal']:0.3g} at {optim_status['time']}\n"
        )
        console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file")
    parser.add_argument(
        "-w",
        "--watch",
        type=int,
        default=0,
        help="Update the table every WATCH seconds.",
    )
    args = parser.parse_args()

    try:
        with open(args.log_file) as file:
            log = hjson.load(file)
        console = Console()
        if args.watch:
            while True:
                show_table(log, console)
                time.sleep(args.watch)
        else:
            show_table(log, console)
    except FileNotFoundError:
        print("Logfile not found. Quiting...")
