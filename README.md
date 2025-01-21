# Hereditary_disc_cup_seg
Code to download publicly available datasets on disc and cup segmentation from Color Fundus Photos.

## Dataset

### Chaksu
This dataset includes 1,345 retinal fundus images stored in JPEG/PNG format. Five expert ophthalmologists provided manual OD and OC annotations and a decision on whether the subject is glaucoma suspect or not. The entire database of 1345 fundus images is divided into training and test subsets comprising 1009 images and 336 images, respectively. The train and test subsets are approximately in the ratio of 3:1. The database provides information about OD and OC height/width/area, and neuroretinal rim area leading to the computation of clinically relevant glaucoma parameters such as vertical cup-to-disc ratio (VCDR), horizontal cup-to-disc ratio (HCDR), and area cup-to-disc ratio (ACDR) from the experts’ annotations.

Available to download as a zip file (10.5 GB) at
> https://figshare.com/articles/dataset/Ch_k_u_A_glaucoma_specific_fundus_image_database/20123135

Download with
`wget https://figshare.com/ndownloader/articles/20123135/versions/2`

This dataset includes images from three devices: Bosch, Forus, and Remidio.

CFP images for each device can be found at
> \<DATA FOLDER>/Train/1.0_Original_Fundus_Images/\<DEVICE>

Ground truth annotations can be found at
> \<DATA FOLDER>/Train/6.0_Glaucoma_Decision/Glaucoma_Decision_Comparison_\<DEVICE>_majority.csv

Specifically, the ground truth labels used in this codebase are "Majority Decision" (called "Glaucoma Decision" in the test set for Forus and Remidio).

## Code

The code will train a ResNet-18 model to predict whether an image is suspicious of glaucoma or not.

Training will be done using Federated Learning, specifically, we will use the three devices as the three clients.

The instructions are similar to those found in the xgboost-CLEF repository.

### Before you start
Change the path to the data folder in the `pyproject.toml` file as well as the path to the output folder where you want to save the model checkpoints.

File `requirements.txt` contains all the dependencies.

### Start the superlink
> flower-superlink --insecure

### Start the nodes (clients)
> flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094 --node-config "partition='Bosch'"

> flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9095 --node-config "partition='Forus'"

> flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9096 --node-config "partition='Remidio'"

### Run the experiment
> flwr run . local-deployment --stream