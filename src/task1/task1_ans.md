# Task 1 Assesment: Protien fitness transformer

This README provides a evaluation of the `trainer.py` proof-of-concept. It answers below questions...

### 1. **What works well?** What aspects of the current design are good and should be preserved?
 - **Amino Acid to index mapping**:In `class ProteinDataset` the `aa_to_idx` dictionary correctly identifies the 20 standard amino acids and a padding token (-).
 - **Data Normalization**: In `class Trainer` the `load_data` function or method, it calculate `mean` and `std` only on training data set, then applies those values to test dataset. This is Standard Normalization works well to keep datasets clean(no mixing of test and train sets).
 - Using `Dataset` and `DataLoader` classes from Pytorch lib, helps with paralell data loading and GPU acceleration.
 - In `class Trainer` the `train` method, training and validation works well. saves `best_model.pt` only when `val_loss` improves.
 - In `class Trainer` the `evaluate` method, calculating MSE, MAE, R2 (Coefficient of Determination) give multi-dimensional view on performance. Saving results to a `results.json` helps that eval data is captured in a machine readable format for later experimentation or testing.
 - visulization with training curves and scatter plot give quick overview with model learning.

### 2. **What are the main issues?** Identify the key problems that would make this design difficult to scale to the larger project. Focus on the most important issues rather than minor style points.
 - In `class Trainer` it is handling `evaluate`, `train`, `load_data` and initialization of transformer, looks like a monolithic design.
 - Hyperparameters such as (`d_model`, `nhead`, `num_layers`, `lr`, `batch_size`), and Splitting data are hardcoded in architecture.
 - In `class Trainer` the `load_data` function has default file path.
 - Saved files will be overwritten when `trainer.py` runs agains and saved to root_dir.
 - Missing logger, project strcuture and modularity of code.
 - lacking virtual environment and package and project manager.
 - Based on run model begins to overfit significantly after Epoch 9 (Train Loss: 0.87 vs Val Loss: 1.17).

### 3. **How would you restructure this?** Describe at a high level how you would refactor the code to better support the team's goals. You don't need to rewrite the code, but be specific about the design changes you'd make.
 - I'd start with using `uv` Rust-based python package and project manager, which is extremely fast. Creates `.toml` and initializes `.git`
 - Create a virtual env with `uv add -r requirements.txt` which add abstract dependencies to `.toml` automatically and creates `uv.lock` which is helpful in cross-platform, capturing dependencies for all operating systems and architectures.
 - Other teams can use `uv sync` to installs dependencies from `uv.lock`.
 - Write a composite [template.py](template.py) which create ***Production-Grade ML Project Structure***, looks like:
 **📂 Structural Breakdown**
 ```bash
 📦root_dir
 ┣ 📂config
 ┃ ┗ 📜config.yaml.             #Stores all file paths
 ┣ 📂frontend                   #User Interface
 ┃ ┣ 📂static
 ┃ ┃ ┣ 📜home.css
 ┃ ┃ ┗ 📜main.js
 ┃ ┗ 📜index.html
 ┣ 📂logs
 ┃ ┗ 📜running_logs.log         #Keep Experimental logs
 ┣ 📂research                   #Experimental Sandbox
 ┃ ┣ 📜01_data_ingestion.ipynb  #Experimenting with downloading or loading
 ┃ ┣ 📜02_data_transform.ipynb  # Experimenting Data tranformations
 ┃ ┣ 📜03_model_building.ipynb  # Designing the Model architecture
 ┃ ┣ 📜04_model_training.ipynb  #Testing the training loop on a small subset of data.
 ┃ ┗ 📜05_test.ipynb            #Running the inference tests and visualizations
 ┣ 📂outputs                    # Save outputs such as .json/.csv
 ┣ 📂models                     #Save models such as .pt/.pkl
 ┣ 📂src
 ┃ ┣ 📂test2
 ┃ ┃ ┣ 📂components
 ┃ ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┃ ┣ 📜data_convert.py
 ┃ ┃ ┃ ┣ 📜data_ingestion.py    #Downloads and extracts your datasets.
 ┃ ┃ ┃ ┣ 📜data_transform.py    #Handles normalization, resizing, transformations
 ┃ ┃ ┃ ┣ 📜model_build.py       #Defines the Network architecture/ Transformer models.
 ┃ ┃ ┃ ┣ 📜model_diagnosis.py   #Contains Precision, Recall, JSON logs functions
 ┃ ┃ ┃ ┣ 📜model_train.py       #Contains the training loop and loss functions
 ┃ ┃ ┃ ┗ 📜test.py              #Runs the final inference
 ┃ ┃ ┣ 📂config
 ┃ ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┃ ┗ 📜configuration.py     #The logic that reads 'config.yaml' and prepares the paths for the components.
 ┃ ┃ ┣ 📂constants
 ┃ ┃ ┃ ┗ 📜__init__.py          #Keep constants initialized
 ┃ ┃ ┣ 📂entity
 ┃ ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┃ ┗ 📜config.py            #Defines Dataclasses. This ensures input/output datatypes
 ┃ ┃ ┣ 📂pipeline               #scripts simply call the components in order
 ┃ ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┃ ┣ 📜stage1.py
 ┃ ┃ ┃ ┣ 📜stage2.py
 ┃ ┃ ┃ ┣ 📜stage3.py
 ┃ ┃ ┣ 📂utils
 ┃ ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┃ ┗ 📜common.py            #Handles methods which are reusable in different modules
 ┃ ┃ ┗ 📜__init__.py
 ┣ 📜.gitignore
 ┣ 📜.python-version            # Project python version track-list
 ┣ 📜README.md                  # Project Information and instructions
 ┣ 📜app.py                     #This connects your AI backend to your frontend.
 ┣ 📜dvc.yaml                   #Data Version Control.
 ┣ 📜main.py                    #The Orchestrator. Runs all the pipeline stages in order.
 ┣ 📜params.yaml                #Stores ML hyperparameters (Learning rate, Patch size, Epochs, Batch size).
 ┣ 📜pyproject.toml             # dependencies list manager
 ┣ 📜requirements.txt           # dependency script
 ┣ 📜setup.py                   #Helpful if you want to change your project into package
 ┣ 📜template.py                #A script you ran once to build this entire folder structure automatically.
 ┗ 📜uv.lock                    #Ensures every computer running this has the exact same libraries.
 ```
 **This structure give full modularity and helps other team to implement, scale for large project and find errors easily**
 - Save models and outputs in a folder with unique names with timestamps.
 - Implement `early stopping`, `dropouts` for handeling model overfit. Determine the  `max_len` from the current dataset dynamically.
 - Keep parameters with results output to understand and keep track of experiments.
 - This [Github repo](https://github.com/captainKLSH/HHMI_Janelia_AI_Engineer) follows a modular architecture, similar to proposed one that separates configuration, data logic, and model execution.