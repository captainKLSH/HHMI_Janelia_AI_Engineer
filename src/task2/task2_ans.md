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

Choosing chunk is essential cause initial chunks such as s0,s1 has higher resolution as we go go futher quality decreases but s0 has more voxel volume than subsequent ones. [Read about datasets here](https://open.quiltdata.com/b/janelia-cosem-datasets).

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

### 1. *Patch size selection*: Which patch size is best suited to capture mitochondrial ultrastructure in the embeddings? Justify your choice.

- ***16x16*** (dictated by the specific DINOv3 ViT-Small plus) we can see this in [model summary](../../OUTPUT/model_summary.txt) generated as kernel_size during model build.
- When analyzing Electron Microscopy (EM) datasets, mitochondrial ultrastructures such as cristae folds, the mitochondrial matrix, and the double-membrane boundary—are exceptionally fine, often spanning just a few pixels at standard nanometer-per-voxel resolutions like (8, 8, 8) voxel size (nm) for Mouse Liver set.
- Manually we cannot change the patch dimension of a pre-trained model.
- But we can work around by choosing input image pefectly matching patch size. E.g: patch size here is 16, so find a image resolution which is perfectly divisible by 16 without reminders.
- To increase patch size we down sample image before sending through model, for smaller patch size we up sample the image.
- Other options is to add padding to increase image size artificially using this principle $\frac{(Input size+(2*padding)-patch size)}{stride}+1$
- We can use overlapping patches by changing stride using $\frac{inputsize-patchsize}{stride}+1$

### 2. *Dense embeddings:* Propose a method for obtaining dense, per-pixel (or per-voxel) embeddings rather than per-patch embeddings. Implement your proposed method and compute dense embeddings for both datasets.

- ***Proposed Method: Token Reshaping and Bilinear Interpolation***
- Vision Transformers like DINOv3 process inputs as discrete patches, inherently outputting a flat 1D sequence of embeddings rather than a dense image map. To achieve dense, per-pixel (or per-voxel) embeddings required for precise semantic retrieval, we must map these patch level features back to the original image coordinate space.
- The proposed method involves intercepting the model's final hidden state, filtering non-spatial data, and mathematically upsampling the remaining tokens.
- We use our model inputs during experimentation input size is *(448,448,448)*, patch size is *16* and model has *384* embeddings these can be interpreted while building model and summary is generated along with it. You can use different input size from [input_size_test](../../params.yaml) from `params.yaml` to see data flow and parameters.
- Our 448 pixel size generates ($\frac{448}{16} = 28$) 28 x 28 patches, Flating it we get 784 vector with 1 [CLS] token and 4 register token.
- **Tokenization**: we have 789 sequence each with 384 embedding keeping pixel co-ordinates in memory with RoPE(**Ro**tary **P**ositional **E**mbeddings )
- These passes through 12 dense tranformer layer, updating 384 embedding with description and context awareness.
- At the end, we keep the 784 pixels + 5 Tokens, 784 pixels is interpolated back to original image size ($784$ --> $28*28$ --> $448*448$)
- ***We have a high-resolution map where every single pixel has its own brilliant 384-number biological fingerprint!***

# Task 3 — Embedding-Based Retrieval & Visualization


