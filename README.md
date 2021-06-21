# PACS_Sampling_Experiments
Repository for running sampling experiments on PACS. 

Example dockerfile to create a container that can run this code:

```docker run -it -v /data/PACS_Dataset:/data/PACS_Dataset --gpus device=6 -v /data/matthew.almeida001/PACS_Sampling_Experiments:/data/matthew.almeida001/PACS_Sampling_Experiments --name pytorch-pacs-sampling matthewalmeida/pytorch-xian```