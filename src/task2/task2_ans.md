# AI Engineer — Microscopy Image Analysis (MIA-AI)



## Take-Home Technical Challenge
### Background

3D electron microscopy (EM) datasets provide rich structural information about brain wiring. Segmenting neurons and their ultrastructure, particularly organelles such as mitochondria, is a key challenge in building connectome datasets. This challenge explores working with state-of-the-art EM image data alongside modern self-supervised learning (SSL) models, specifically DINO.

## Task 1 — Data Acquisition

### 📥 How Data Ingestion Works

The `DataIngestion` script acts as a efficient delivery service for the AI pipeline. The microscopic cell datasets from [OpenOrganelle](https://openorganelle.janelia.org/organelles/mito) are incredibly large—often terabytes in size and stored on cloud servers (Amazon S3). 

Instead of forcing the computer to download the entire massive dataset, this code smartly targets, slices, and extracts only the specific 3D microscopic chunks required for the task.

**1. Setting the Target (Initialization)**
When the process begins, the script reads a set of instructions [config.yaml](../../config/config.yaml). This configuration acts like a map, providing the web address of the cloud storage, the destination folder on the local computer, Specify chunk and the exact X, Y, and Z coordinates from [params.yaml](../../params.yaml) of the 3D cell block we want to extract.

**2. The Download (`download_file`)**
* **Checking Local Inventory:** Before using internet, the code first checks if the requested cell chunk has already been downloaded to the computer. If it is there, it skips the download process entirely to save time.
* **Connecting to the Source:** If the file is needed, the script connects directly to the public cloud database.
* **Precision Slicing:** Rather than downloading the whole dataset and cropping it later, the code applies our X, Y, and Z coordinates directly to the cloud server using [fib_semtools](https://github.com/janelia-cellmap/fibsem-tools). It acts like a digital cookie-cutter, isolating just the 3D area we care about.
* **Saving for Speed:** It pulls that specific chunk into memory and saves it locally as a `.npy` (NumPy) file. This specific file format is optimized for speed, allowing the DINOv3 AI model to read the image data almost instantly later in the pipeline.
* **Logging:** It records the metadata and details of the download into a text file so we have a permanent record of what was extracted.

**3. Inspecting the Delivery (`data_dim`)**
Once the file is safely on the computer, the script runs a quick inspection. It peeks at the file without fully loading the massive image into the computer's active memory. It calculates and logs:
* **Shape:** The physical dimensions of the 3D block.
* **Total Elements:** The exact number of data points (voxels) making up the image.
* **Memory Size:** How many Megabytes the chunk occupies.

## Task 2 — Feature Extraction with DINO


