import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "task2"

logging_content = f"""import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("{project_name}Logger")
"""

main_logger=f"""from src.{project_name} import logger

logger.info("Starting {project_name}.... ... .. . .")
"""
require_list="""# --- Core AI & Data Science ---
pandas                  # The spreadsheet manager
numpy                   # The math engine
matplotlib              # The graphing tool
joblib                  # The "Deep Freezer" for binary files

# --- Advanced Utilities ---
python-box              # Powers ConfigBox (Dot-notation access)
ensure                  # Powers @ensure_annotations (Type safety)
pyYAML                  # Powers read_yaml (Config management)
tqdm                    # Powers your Progress Bars
scipy                   # Advanced math for image processing

# --- Backend & Web ---
fastapi                 # The high-speed kitchen (API)
uvicorn                 # The engine that runs FastAPI
python-multipart        # Allows users to upload CT scan files
Jinja2                  # The HTML menu designer

# --- MLOps & Tracking ---
dvc                     # Data versioning (GitHub for big files)
mlflow                  # The digital lab notebook
dagshub                 # The central hub for your experiments
python-dotenv           # Keeps your secret keys safe
"""
constant_fill="""from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
"""
file_contents = {
    f"src/{project_name}/__init__.py": logging_content,
    "main.py": main_logger,
    "requirements.txt": require_list,
    f"src/{project_name}/constants/__init__.py":constant_fill,
}

list_of_files = [
    ".github/workflows/.gitkeep",
    "main.py",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "frontend/index.html",
    "frontend/main.js",
    "frontend/home.css"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    # Split the path into (Folder, Filename)
    filedir, filename = os.path.split(filepath)

    # 1. Handle Folder Creation
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # 2. Handle File Creation (Works for files in folders AND main directory)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            content = file_contents.get(filepath.as_posix(), "")
            f.write(content)
            
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists and is not empty")