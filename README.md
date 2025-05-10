## Train and Test Overexpression models

It is sufficient to run the scripts `train.py` and `test.py`. To train the DSMIL and DS_ABMIL models, it is necessary to run the corresponding files within the `MIL` folder.


## How to Run Over-expression demo

Launch one of the following files with access to an NVIDIA GPU:

- `attention_map.py`
- `old_attention_map.py`

Both files provide predictions for the overexpression of BRCA1. The first file provides an attention map with internal attention for each patch, while the second file only provides attention on the patches.
Both scripts expect the following command-line arguments:

- `-m` to specify the model ID to test. The models must be located in the current folder and have the following name format: `model_weights_X.pth`, where X is the ID. The default is `0`.
- `-s` to specify the patient ID to test. The default is `'TCGA-E2-A155'`.

## Train Overall Survival models

#### MCAT

- **With CAB**  
  ```bash
  sbatch train_MCAT_cab.sh
  ```
- **Without CAB**  
  ```bash
  sbatch train_MCAT.sh
  ```
#### MCAT (BRCA1 and BRCA2)

- ```bash
  sbatch train_sh/train_MCAT.sh
  ```
#### SurvPath
Before running SurvPath, you must first navigate to the appropriate directory:
```bash
cd /work/ai4bio2024/brca_surv/survival/SurvPath
```
Then run:
- **With CAB**  
  ```bash
  sbatch train_SurvPath_cab.sh
  ```
- **Without CAB**  
  ```bash
  sbatch train_SurvPath.sh
  ```
  

## How to Run Overall Survival demo

### 1. Submit the Demo Job

To start the demo, simply run the following command from the project directory:

```bash
sbatch demoOS.sh
```

This will submit a job to the cluster which runs the demo server (e.g., a Streamlit app) on a specific node.
### 2. Set Up Port Forwarding in VSCode

Once the job starts, check the node name where it is running (e.g., ajeje or pippobaudo). Then, in VSCode:

1. Go to the Port Forwarding tab.
2.  Add a forwarded port with this format:
```bash
node_name:8501
```
Example:
```bash
ajeje:8501
```
  Make sure port ```8501``` is open and not used by other processes.

### 3. Access the Demo Locally

After port forwarding is set up, open your browser and navigate to:

http://localhost:8501

You should now see the current interface: ![](img/demo1.png)

### 4. Select the Model Weights and Enter Patient ID

Once the demo interface is visible at `http://localhost:8501`, follow these steps:

1. **Select the MCAT weights**  
   At the top of the interface, you'll find a file selector or dropdown to choose the model weights.  
   You can download the available weight files from the following link:

   👉 [Download model weights from Google Drive](https://drive.google.com/drive/folders/1AEz8LCSWBxUGjOxhNfpERG4iOjh6uh1H?usp=drive_link)


2. **Enter the Patient ID**  
   Below the model selection, there will be a field where you can input a **Patient ID**.  
   This ID should match one of the available cases in the dataset used by the model.

3. **Run the analysis**  
   After selecting the weights and entering the patient ID, click on "Run Prediction" in order to display the results.

> ⚠️ If no weights are selected or the patient ID is invalid, the application may show an error or fail to produce results.

![](img/demo2.png)

### 5. Results
- Patient ID: TCGA-E2-A155 ![](img/demo3.png)
- Patient ID: TCGA-EW-A2FW ![](img/demo4.png)
  
## Overall Survival evaluation
This notebook contains the results exactly as presented in the paper.

📍 Path to the notebook:  
`/work/ai4bio2024/brca_surv/survival/evaluation.ipynb`
