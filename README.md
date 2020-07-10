## Code structure
* `./partition/*` - Partition code (geometric partitioning and superpoint graph construction using handcrafted features)
* `./supervized_partition/*` - Supervized partition code (partitioning with learned features)
* `./learning/*` - Learning code (superpoint embedding and contextual segmentation).

## Disclaimer
Our partition method is inherently stochastic. Hence, even if we provide the trained weights, it is possible that the results that you obtain differ slightly from the ones presented in the paper.

## Requirements 
*0.* Download current version of the repository. Download the [cut pursuit](https://github.com/loicland/cut-pursuit) module used in `/partition`. <br>

*1.* Install [PyTorch](https://pytorch.org) and [torchnet](https://github.com/pytorch/tnt).
```
pip install git+https://github.com/pytorch/tnt.git@master
``` 

*2.* Install additional Python packages:
```
pip install future python-igraph tqdm transforms3d pynvrtc fastrlock cupy h5py sklearn plyfile scipy
```

*3.* Install Boost (1.63.0 or newer) and Eigen3, in Conda:<br>
```
conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; conda install -c r libiconv
```

*4.* Make sure that cut pursuit was downloaded. Otherwise, clone [this repository](https://github.com/loicland/cut-pursuit) or add it as a submodule in `/partition`: <br>
```
cd partition
git submodule init
git submodule update --remote cut-pursuit
```

*5.* Compile the ```libply_c``` and ```libcp``` libraries:
```
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
cd ..
cd cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```
*6.* Install [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)

The code was tested on Ubuntu 18.04 with Python 3.6 and PyTorch 1.4.

### Troubleshooting

Common sources of errors and how to fix them:
- $CONDAENV is not well defined : define it or replace $CONDAENV by the absolute path of your conda environment (find it with ```locate anaconda```)
- anaconda uses a different version of python than 3.6m : adapt it in the command. Find which version of python conda is using with ```locate anaconda3/lib/libpython```
- you are using boost 1.62 or older: update it
- cut pursuit did not download: manually clone it in the ```partition``` folder or add it as a submodule as proposed in the requirements, point 4.
- error in make: `'numpy/ndarrayobject.h' file not found`: set symbolic link to python site-package with `sudo ln -s $CONDAENV/lib/python3.7/site-packages/numpy/core/include/numpy $CONDAENV/include/numpy`


