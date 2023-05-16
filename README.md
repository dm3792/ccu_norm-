Modelccu.py contains the main model and calls the data loader, NSIT average precision scorer and combines the ldc data from data loader with Yi’s data for norms in the ldcNormsCombine.py file.

How to run the code?
On the server choose a GPU using export CUDA_VISIBLE_DEVICES=x.
To help with the choosing process you can use the nvidia-smi command to check the available GPUs.
If the python version doesn’t support the scorer, set up and activate a conda environment with python version
Clone the repository
Run the command (or script) : python3 modelccu.py 
Various arguments that can be added in the command are as such:
–device = cuda (to use GPU for faster processing)
–batch-size = to set the batch size
–lr = set learning rate
–include-utterance = if used, the utterances are included
–confident-only = only use confident norms
–regularisation = options l1, l2, dropout
–downsample = specify the ratio
–classifierlayers = number of layers in the classifier


Variants 
The script for commands to run different variants of the model are present in test.sh and test2.sh. 
