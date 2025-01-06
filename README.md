# VisEdit

<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="figures/img_attribution.svg" alt="Attribution" style="width: 70%; margin-right: 5px;">
  <img src="figures/img_method.svg" alt="Method" style="width: 27%;">
</div>


Source code for AAAI 2025 paper [*Attribution Analysis Meets Model Editing: Advancing Knowledge Correction in Vision Language Models with VisEdit*.](https://arxiv.org/abs/2408.09916/)

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



# Citation
Please cite our paper if you use VisEdit in your work (The AAAI citation is not yet available).
```bibtex
@article{DBLP:journals/corr/abs-2408-09916,
  author       = {Qizhou Chen and
                  Taolin Zhang and
                  Chengyu Wang and
                  Xiaofeng He and
                  Dakan Wang and
                  Tingting Liu},
  title        = {Attribution Analysis Meets Model Editing: Advancing Knowledge Correction
                  in Vision Language Models with VisEdit},
  journal      = {CoRR},
  volume       = {abs/2408.09916},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2408.09916},
  doi          = {10.48550/ARXIV.2408.09916},
  eprinttype    = {arXiv},
  eprint       = {2408.09916},
  timestamp    = {Fri, 15 Nov 2024 07:55:45 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2408-09916.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


# Reference
[1] Can We Edit Multimodal Large Language Models?



