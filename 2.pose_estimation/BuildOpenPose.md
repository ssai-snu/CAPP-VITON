# Build openpose

# Instruction : 
> https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source

## build openpose from source
>
>>git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
>>
>>mv openpose openpose_build
>>
>>cd openpose_build/
>>
>>git submodule update --init --recursive --remote
>>
>>mkdir build/
>>
>>cd build/
>>
>>cmake-gui ..
>
```bash
make -j`nproc`
```
