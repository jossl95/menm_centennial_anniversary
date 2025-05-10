from pathlib import Path
import subprocess
from tqdm.notebook import tqdm

CODEDIR = Path('code')

def run(runfile):
    result = subprocess.run(["python", runfile], capture_output=True, text=True)


def main():
    items = sorted(CODEDIR.glob('*.py'))
    files = [file for file in items if '._' not in str(file)]
    [run(file) for file in tqdm(files[4:9])]

if __name__ == "__main__":
    main()
