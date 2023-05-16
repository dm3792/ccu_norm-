Modelccu.py contains the main model and calls the data loader, NSIT average precision scorer and combines the ldc data from data loader with Yi’s data for norms in the ldcNormsCombine.py file.
<br />
<br />
How to run the code?<br />
On the server choose a GPU using export CUDA_VISIBLE_DEVICES=x. <br />
To help with the choosing process you can use the nvidia-smi command to check the available GPUs.<br />
If the python version doesn’t support the scorer, set up and activate a conda environment with correct python version 3.11 <br />
Clone the repository <br />
Run the command (or script) : python3 modelccu.py  <br />
Various arguments that can be added in the command are as such: <br />
–device = cuda (to use GPU for faster processing) <br />
–batch-size = to set the batch size <br />
–lr = set learning rate <br />
–include-utterance = if used, the utterances are included <br />
–confident-only = only use confident norms <br />
–regularisation = options l1, l2, dropout <br />
–downsample = specify the ratio <br />
–classifierlayers = number of layers in the classifier <br />
<br /><br />

Variants <br />
The script for commands to run different variants of the model are present in test.sh and test2.sh. 
