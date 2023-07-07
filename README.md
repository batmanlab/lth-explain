## Environment setup

Create the environment from the environment.yml file:

 ```bash
  conda env create -f environment.yml
 ```

## Dataset

#### CUB-200

* Download the CUB-200 dataset from the [link](https://www.vision.caltech.edu/datasets/cub_200_2011/)
* Preprocess the noisy concepts in the dataset using the following command:

 ```bash
cd scripts_data
python download_cub.py
```

## Dataset splits

The train-test-val splits of all the datasets are given in the corresponding json files in the `scripts_data` directory.

## Hyperparameters

* For CUB-200, check `config/BB_cub.yml` file.
* For HAM10000, check `config/BB_derma.yml` file

## Steps to reproduce the results

* Prior to start the training process, edit `data_root`, `json_root` and `logs` parameters in the config
  file `config/BB_cub.yaml` to set the path of images, json files for train-test-val splits and the output to be saved
  respectively.
* Prior to follow the steps, refer to `./iPython/Cub-Dataset-understanding.ipynb` file to understand the CUB-200
  dataset. This step is optional.
* Preprocess the noisy concepts as described earlier.
* Follow the steps below for CUB-200 dataset:


#### Step 1: Train and prune

```bash
python main_lth_pruning.py --dataset "cub"
```

#### Step 2: Save the activations as image embeddings to train for CAVs

```bash
python main_lth_save_activations.py --dataset "cub"
```

#### Step 3: Train classifier using the image embeddings to obtain CAVs

```bash
python main_lth_generate_cavs.py --dataset "cub"
```

#### Step 4: Train PCBM

```bash
python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_pcbm.py --dataset "cub"
```

#### Step 5: Get top-3 concepts from PCBM 

```bash
python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_get_concepts.py --dataset "cub"
```

#### Step 6: Compute Grad-CAM based saliency maps for the local explanations

Edit `labels_for_tcav` parameter in the file `config/BB_cub.yaml` for the desired class label to generate the Grad-CAM
saliency maps. By default, we generate the saliency map for the 2nd image of the desired class in the test-set.

```bash
python main_heatmap_save.py --config "config/BB_cub.yaml"
```

#### Step 7: Generate plots

```bash
./iPython/Analysis-CUB_Test-GradCAM.ipynb
```

## Bash scripts

All the bash scripts to follow steps are included in `./bash_script` file.
