import asyncio
import os
import shutil

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def create_trial_notebook(trial_dir, notebook_path, status_queue=None, trial_num=None):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "trial_analysis.ipynb")
    shutil.copy(template_path, notebook_path)

    executed_notebook_path = notebook_path.replace(".ipynb", "_executed.ipynb")
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        with open(notebook_path) as f:
            nb = nbf.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": trial_dir}})
        with open(executed_notebook_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        os.remove(notebook_path)
    except Exception as e:
        if status_queue is not None and trial_num is not None:
            status_queue.put((trial_num, f"Trial {trial_num} | Notebook error: {e}"))
        else:
            print("Error during notebook execution:", e)


def create_aggregate_notebook(unique_results_dir, notebook_path):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", "aggregate_analysis.ipynb")
    shutil.copy(template_path, notebook_path)

    executed_notebook_path = notebook_path.replace(".ipynb", "_executed.ipynb")
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        with open(notebook_path) as f:
            nb = nbf.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": unique_results_dir}})
        with open(executed_notebook_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("Executed summary notebook saved at", executed_notebook_path)
        os.remove(notebook_path)
    except Exception as e:
        print("Error during notebook execution:", e)

    print("Summary notebook creation process complete.")
