# PHORECAST - Public Health Outreach REceptivity and CAmpaign Signal Tracking

<!-- ![Project Screenshot]() -->

<p align="center">
  <img src="teaser.png" width="" height="">
</p>

## Description
This repository contains the PHORECAST dataset alongside the code for our paper. 

PHORECAST maps demographics and personalities into their corresponding reactions to public health marketing campagins.
The processed dataset can be found at huggingface.co/datasets/tomg-group-umd/PHORECAST. 
## Code Details

We consider there to be three phases to the codebase for this project: 
1. Data Processing: We provide the methods used to create our training/validation set in `./processing`:
    - The data procsesing step is the most intricate process, since there are many choices to be made here (feature randomization, training on all features or only some, defining the validation split as either a set \% or a more stratified sampling method using the demogrphic groups.) 
    = Please run `python data_prep.py` to prepare the training/validation splits, as used in the paper (feature randomization and hold out strategy across gender, religion and race). Feel free to experiment with the randomization of features and what we use for training or validation.

2. Training scripts are found under `./training_scripts`
    - We use Unsloth for training. Our script supports Pixtral, Gemma, Llama, Llava, and Qwen. If you want to start training, run the following command: ```./train.sh gemma {path_to_dataset} {path_to_output}```
    
3. All evaluation scripts are found under `./eval`
    - Once we have a trained model, we can run inference on the validation split using `run_inf.sh` and evaluate the model accuracy or STS similarity using `eval.sh "$DATA_FILE" "$THRESHOLD"`. 
