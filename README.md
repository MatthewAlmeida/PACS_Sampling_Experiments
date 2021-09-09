# PACS_Sampling_Experiments
Repository for running sampling experiments on PACS, a benchmarking dataset for Domain Generalization in image classification. Data files are available on Google Drive, open to the internet at [this link](https://drive.google.com/drive/u/0/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ), last tested 6/28/21.

Example docker command to create a container that can run this code (uses the dockerfile and requirements.txt from  [this repository](https://github.com/MatthewAlmeida/Pytorch-dockerfiles/tree/main/xian)):

```docker run -it -v /data/PACS_Dataset:/data/PACS_Dataset --gpus device=6 -v /data/matthew.almeida001/PACS_Sampling_Experiments:/data/matthew.almeida001/PACS_Sampling_Experiments --name pytorch-pacs-sampling matthewalmeida/pytorch-xian```

Additionally, you may want to specify the following environment variables in a .env file (defaults provided):

```
LOG_DIR=lightning_logs
CHKPT_DIR=checkpoints
TENSORBOARD_PORT=6006
PACS_HOME=/data/PACS_Dataset
```

```generate_seeds.sh``` creates a text file of random seeds for the experiments. ```run_experiments.sh``` loops through the random seeds (renamed to ```seeds_final.txt``` to prevent accidental overwrite) and runs one experiment for each seed. Storage warning: each experiment saves a model checkpoint that is ~134MB, so the full 200-model experiment will leave behind ~26 GB of checkpoint data in the provided directory.

---

These experiments were inspired by a recent attempt to reproduce the PACS benchmarking results in a recent Domain Generalization paper. When looking through implementation details in that paper's repository, it became clear that their models were trained by drawing entire batches of data at once from the available training domains, rather than shuffling the whole training set together and drawing batches from that mix. That is, training was done by randomly drawing, for example, a full batch of photographs, then a full batch of sketches, then a full batch of paintings, and so on, rather than combining all of the paintings and sketches and photographs and drawing batches that contain some of each.

This should have meaningful consequences when viewed in terms of SGD: rather than updating parameters based on stochastic estimates of the full training set gradient, this implementation uses a random sequence of estimates of the gradients with repect to each domain. 

Given that the improvements provided in the paper were incremental (which is not meant dismissively: most results are!) and reported a relatively small effect size over completing methods, experience with normal variation in deep learning training processes (over hyperparameters, data preprocessing methods, random seeds, etc) suggests that the variation in performance due to this choice of sampling implementation may be on the order of the effect size reported in the paper. 

This repository exists to test if this is the case.  Using a pretrained ResNet-18 as the base model (weight initialization is held constant), 100 models are trained under each of the two sampling regimes: single-domain batches and random batches. Power analysis offers that 63 models of each type are adequate to detect a 2% difference in means (at the 0.05/ 0.8 level) assuming the models have an average performance of 50% accuracy with a 4% standard deviation. A sample size of 100 per setting is a compromise between precision and computing time.

The first set of results will be performed using "art_painting" as the test set and "photo", "sketch", and "cartoon" as the training data. Results will be compiled in a Jupyter Notebook and reported here when the experiments conclude.
