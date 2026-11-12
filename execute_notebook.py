from pathlib import Path
import argparse
import json
import sys
import platform
import nbformat
from nbclient import NotebookClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("number", type=int)
    parser.add_argument("--mode", choices=["full", "cpu", "data"], default="cpu")
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / f"PZ_{args.number}" / f"PZ_{args.number}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    notebook.metadata.language_info.version = platform.python_version()
    allowed = {"setup", "definition", "unit", "local_report"}
    if args.mode == "data":
        allowed.add("data")
    client = NotebookClient(notebook, timeout=None, kernel_name="python3", resources={"metadata":{"path":str(path.parent)}})
    sys.stdout.reconfigure(encoding="utf-8")
    with client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            tags = set(cell.metadata.get("tags", []))
            if "install" in tags:
                continue
            if args.mode != "full" and not (tags & allowed):
                continue
            print(f"{path.stem}: cell {index} {tags}", flush=True)
            try:
                client.execute_cell(cell, index)
            except Exception as error:
                nbformat.write(notebook, path)
                outputs = cell.get("outputs", [])
                errors = [output for output in outputs if output.output_type == "error"]
                print(errors[-1].get("evalue",str(error)) if errors else str(error), flush=True)
                raise SystemExit(1)
            nbformat.write(notebook, path)
    print("Saved", path, flush=True)

if __name__ == "__main__":
    main()
