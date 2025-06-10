# PCD_SM-Fluid-Pred

### Installation details

For export:
conda env export | grep -v "^prefix: " > environment.yml

For import:
conda env create -f environment.yml

conda remove -n SM-Fluid-Pred --all

pdebench & theWell need python 3.10

python 3.10

PipLy

torch 1.13.0
pdebench
the_well
the_well[benchmark]
jupyter
neuraloperator (theWell)


**to be excluded for code publishing:** 

all from cluster dir
WandB token 
cluster training scripts
-> refactor training
find references to my computer paths, delete ALL and associates
