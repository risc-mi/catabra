import os
from pathlib import Path
from typing import Union, List

import json
import glob


class NBContent:
    cells = 'cells'
    meta = 'metadata'
    post_execute = 'post_cell_execute'
    type = 'cell_type'
    tags = 'tags'


def check_if_notebook(file_path: Union[str, Path]):
    if not os.path.isfile(file_path) or not file_path.endswith('ipynb'):
        raise FileNotFoundError(file_path + "is not a valid Jupyter Notebook.")


def mock_input(file_path: Union[str, Path], mock: str = 'y'):
    """
    Ads a cell to the given notebook that overwrites the CaTabRa prompt functionality with a function that always
    returns the mock value. This is needed so that notebooks can be executed automatically without user input.

    Parameters
    ----------
    file_path: str | Path
        Path to the notebook file. File extension must be ".ipynb".
    mock: str
        Value to return as the mocked input.
    """
    check_if_notebook(file_path)

    with open(file_path, 'r') as file:
        content = json.load(file)
        cells = content[NBContent.cells]
        cells = [
                    {
                        "cell_type": "code",
                        "execution_count": 0,
                        "id": "0",
                        "metadata": {
                            "tags": ["patch"]
                        },
                        "outputs": [],
                        "source": [
                            "def prompt(msg: str, accepted=None, allow_headless=True):\n",
                            "    print(msg)\n",
                            "    return '" + mock + "'\n",
                            "\n",
                            "import catabra\n",
                            "import catabra.util\n",
                            "import catabra.util.logging\n",
                            "catabra.util.logging.prompt = prompt"
                        ]
                    }
                ] + cells
        content[NBContent.cells] = cells

    with open(file_path, 'w') as file:
        json.dump(content, file)


def remove_mocked_input(file_path: Union[str, Path]):
    """
    Removes any patched mock input cells from the notebook.

    Parameters
    ----------
    file_path: str | Path
        Path to the notebook file. File extension must be ".ipynb".
    """
    check_if_notebook(file_path)

    with open(file_path, 'r') as file:
        content = json.load(file)
        cells = content[NBContent.cells]
        print(cells[0])
        while NBContent.tags in cells[0][NBContent.meta] and \
                'patch' in cells[0][NBContent.meta][NBContent.tags]:
            cells = cells[1:]
        print(cells[0])
        content[NBContent.cells] = cells

    with open(file_path, 'w') as file:
        json.dump(content, file)


def process_notebooks(nbs: Union[str, List[str]], mock: str = 'y', unpatch: Union[int, bool] = False):
    if os.path.isdir(os.path.join(nbs)):
        nbs = glob.glob(os.path.join(nbs) + '/*.ipynb')
    elif isinstance(nbs, str):
        nbs = [nbs]
    for nb in nbs:
        if unpatch:
            remove_mocked_input(nb)
        else:
            mock_input(nb, mock or 'y')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nbs", help="Jupyter notebooks file, list of such files or directory containing"
                                    "multiple notebooks")
    parser.add_argument("-m", "--mock", help="Input mock value. Defaults to \"y\".")
    parser.add_argument("-u", "--unpatch", help="Remove the patched line from the notebook(s) <0,1>.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_notebooks(args.nbs, args.mock, args.unpatch)
