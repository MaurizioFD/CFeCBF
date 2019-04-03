# CFeCBF

This repository contains the core model we called "**Collaborative filtering enhanced Content-based Filtering**" published in our UMUAI article "**Movie Genome: Alleviating New Item Cold Start in Movie Recommendation**"

The article is accessible [HERE](https://doi.org/10.1007/s11257-019-09221-y), the dataset is accessble [HERE](https://mmprj.github.io/mtrm_dataset/index).

Please cite our article if you use this repository or algorithm.


```
@Article{Deldjoo2019,
author="Deldjoo, Yashar
and Ferrari Dacrema, Maurizio
and Constantin, Mihai Gabriel
and Eghbal-zadeh, Hamid
and Cereda, Stefano
and Schedl, Markus
and Ionescu, Bogdan
and Cremonesi, Paolo",
title="Movie genome: alleviating new item cold start in movie recommendation",
journal="User Modeling and User-Adapted Interaction",
year="2019",
month="Feb",
day="26",
issn="1573-1391",
doi="10.1007/s11257-019-09221-y",
url="https://doi.org/10.1007/s11257-019-09221-y",
note="Source: \url{https://github.com/MaurizioFD/CFeCBF}",
}
```



See run_example.py for an example on how to use the code


## Installation

Note that this repository requires Python 3.6

First we suggest you create an environment for this project using virtualenv (or another tool like conda)

First checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment:

If you are using virtualenv:
```Python
virtualenv -p python3 CFeCBF
source CFeCBF/bin/activate
```
If you are using conda:
```Python
conda create -n CFeCBF python=3.6 anaconda
source activate CFeCBF
```

Then install all the requirements and dependencies
```Python
pip install -r requirements.txt
```

This repository already contains compiled Cython code for Linux and Windows x86. 
In order to compile you must have installed: _gcc_ and _python3 dev_, which can be installed with the following commands:
```Python
sudo apt install gcc 
sudo apt-get install python3-dev
```

At this point you can compile all Cython algorithms by running the following command. The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some warnings. 
 
```Python
python run_compile_all_cython.py
```

