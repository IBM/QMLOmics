# ISMB Tutorial

## Getting Started

### Prerequisites
1. Create the environment from the `env.txt` file:
```
conda create -n "QML" 
pip install -r env.txt
```
* Note: if you receive the error `bash: conda: command not found...`, you need to install Anaconda to your development environment (see "Additional resources" below)
2. Activate the new environment:
```
conda activate QML
```
3. Verify that the new environment was installed correctly:
```
conda env list
```
<!-- * Additional resources:
   * [Connect to computing cluster](http://ccc.pok.ibm.com:1313/gettingstarted/newusers/connecting/)
   * [Set up / install Anaconda on remote linux server](https://kengchichang.com/post/conda-linux/)
   * [Set up remote development environment using VSCode](https://code.visualstudio.com/docs/remote/ssh) -->

<a name="running_comical"></a>
<!-- ### Running QML -->

<!-- [![Notebook Template][notebook]](#running_comical) -->

<!-- 1. Request resources from computing cluster:
```
jbsub -cores 2+1 -q x86_1h -mem 5g -interactive bash
```
OR
Submit your job without the interactive session (shown later).  -->

<!-- 2. Activate the new environment:
```
conda activate QML
``` -->
4. Run QML pipeline:
```
python main.py --file your_file -e 50 -i 10 -p 5 -cv 5
```


### Help
```
python main.py --help
```

## Authors

Contributors and contact info:

* Aritra Bose (a.bose@ibm.com)
