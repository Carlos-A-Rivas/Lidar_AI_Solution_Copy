# Getting Lidar_AI_Solution_Copy up and running

Make sure you recursively clone the git repository:
```bash
git clone --recursive https://github.com/Carlos-A-Rivas/Lidar_AI_Solution_Copy.git
```

Download the models (all models will download from either of the provided links, but **NVBox is easier**):
- [NVBox](https://nvidia.box.com/shared/static/vc1ezra9kw7gu7wg3v8cwuiqshwr8b39)
- [Baidu Drive](https://pan.baidu.com/s/1BiAoQ8L7nC45vEwkN3bSGQ?pwd=8jb6)

Insert the `model.zip` file in the following directory (and switch to the folder in CLI):
```bash
cd Lidar_AI_Solution_Copy/CUDA-BEVFusion/
```
Unzip:
```bash
sudo apt install unzip
unzip model.zip
rm ~/Lidar_AI_Solution_Copy/CUDA-BEVFusion/model.zip
```
The file tree should look like this:
```bash
CUDA-BEVFusion
|-- custom-example
    |-- 0-image.jpg
    |-- ...
    |-- camera_intrinsics.tensor
    |-- ...
    `-- points.tensor
|-- src
|-- qat
|-- model
    |-- resnet50int8
    |   |-- bevfusion_ptq.pth
    |   |-- camera.backbone.onnx
    |   |-- camera.vtransform.onnx
    |   |-- default.yaml
    |   |-- fuser.onnx
    |   |-- head.bbox.onnx
    |   `-- lidar.backbone.xyz.onnx
    |-- resnet50
    `-- swint
|-- bevfusion
`-- tool
```

Install python dependency libraries:
```bash
sudo apt-get update
sudo apt install libprotobuf-dev
sudo apt install python3-pip
pip install onnx
```

__The following steps assume you already have the proper TensorRT version installed.

Modify the TEnsorRT/CUDA/CUDANN/BEVFusion variable values in the tool/environment.sh file
```bash
# change the path to the directory you are currently using
export TensorRT_Lib=/path/to/TensorRT/lib
export TensorRT_Inc=/path/to/TensorRT/include
export TensorRT_Bin=/path/to/TensorRT/bin

export CUDA_Lib=/path/to/cuda/lib64
export CUDA_Inc=/path/to/cuda/include
export CUDA_Bin=/path/to/cuda/bin
export CUDA_HOME=/path/to/cuda

export CUDNN_Lib=/path/to/cudnn/lib

# For CUDA-11.x:    SPCONV_CUDA_VERSION=11.4
# For CUDA-12.x:    SPCONV_CUDA_VERSION=12.6
export SPCONV_CUDA_VERSION=12.8

# resnet50/resnet50int8/swint
export DEBUG_MODEL=resnet50int8

# fp16/int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=custom-example
export USE_Python=OFF
```

Apply the environment to the current terminal.
```bash
. tool/environment.sh
```

## To Build the environment once everything else is setup:

1. Building the models for tensorRT
```bash
bash tool/custom_build_trt_engine.sh
```
2. Compile and run the program
```bash
# Generate the protobuf code
bash src/onnx/make_pb.sh

# Compile and run
bash tool/custom_run.sh
```

## For Python Interface
1. Modify `USE_Python=ON` in custom_environment.sh to enable compilation of python.
2. Run `bash tool/run.sh` to build the libpybev.so.
3. Run `python tool/pybev.py` to test the python interface.

## OTHER TIPS:
- If trying to accomplish things other than what is mentioned in my custom readme, [reference the original readme](https://github.com/Carlos-A-Rivas/Lidar_AI_Solution_Copy/tree/main/CUDA-BEVFusion)
- The repo is setup with the python interface off, but I think turning it on and using `custom_pybev.py` might be the key to getting proper results.
- Look through `make_custom_tensors.py` and `make_custom_tesnors2.py` to sanity check and make sure I entered the matrices correctly. I have been using `make_custom_tensors.py` as the primary file, and `make_custome_tensors2.py` to just test different things.
- I have been working out of the `tool` and `src` folders, but I realized there is a Dockerfile in `~/Lidar_AI_Solution_Copy/CUDA-BEVFusion/bevfusion/docker`; I can't believe I missed this 🤦. This looks like a good spot to poke around, along with `~/Lidar_AI_Solution_Copy/CUDA-BEVFusion/bevfusion/tools`. 
- The `CMakeLists.txt` file in the `~/Lidar_AI_Solution_Copy/CUDA-BEVFusion` folder HAS BEEN EDITED. The original `CMakeLists.txt` file can be found in the original Lidar_AI_Solution repo, and may need to be substituted to get docker to work (I do not know though).
- Do not hesitate to reach out via carivas007@gmail.com after my Caltech email/slack expires with questions! (My Caltech email should be valid for the next couple months.)