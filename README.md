# Setup
1. Please download the E-EVQA and E-IC datasets from the URL provided in [1] and place the related folders in the `data` directory.
2. Please modify the `ROOT_PATH` in `utils/GLOBAL.py` to the absolute path of the current directory, and update `model_path_map` to the absolute paths of each backbone's weights.

# Module Contribution Attribution 
Please run `contribution_module.py`, using Jupyter Notebook would be better for display.

# Visual Representation Contribution Attribution 
Please run `contribution_visual_reps.py`, using Jupyter Notebook would be better for display.

# train VEAD
Please use the following script to train a VEAD:

`python vead_train.py -mn llava -dna EVQA -bs 4 -dvc "cuda:0" -edvc 1 `

# evaluate VEAD
Please use the following script to test VEAD:

`python vead_test.py -mn llava -dn EVQA -dvc "cuda:0" -ckpt [vead_checkpoint_path]`

[1] Can We Edit Multimodal Large Language Models?

