# A100TF2
test A100 TF2 NGC container 20.06-tf2-py3
1.	Call out windows power shell 
2.	Run sft ssh sae-login <-- should get redirected to annex cluster 
3.	Acquire an A100 machine :  srun -p dgxa100 --nodes=1 -n 1  -G 1 --pty bash
4.	Fetch the NGC docker image : singularity build name=tf2-sif docker://nvcr.io/nvidia/tensorflow:20.06-tf2-py3 
5.	Run the singularity container : sigularity shell --nv name=tf2-sif
6.	Clone a git repo : git clone https://github.com/Zenodia/A100TF2.git
7.	Run the script ( the data is baked in ) : python tf2unet.py 
