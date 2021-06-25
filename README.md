# PACS_Sampling_Experiments
Repository for running sampling experiments on PACS, a benchmarking dataset for Domain Generalization in image classification. 

Example docker command to create a container that can run this code (uses the dockerfile and requirements.txt from  [this repository](https://github.com/MatthewAlmeida/Pytorch-dockerfiles/tree/main/xian)):

```docker run -it -v /data/PACS_Dataset:/data/PACS_Dataset --gpus device=6 -v /data/matthew.almeida001/PACS_Sampling_Experiments:/data/matthew.almeida001/PACS_Sampling_Experiments --name pytorch-pacs-sampling matthewalmeida/pytorch-xian```
