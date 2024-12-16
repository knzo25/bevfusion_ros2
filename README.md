
# Build the image

```
docker build --network=host --progress=plain -t bevfusion_docker .
```

# Start the container

```
docker run --net host --gpus all -v $(pwd)/autoware_lidar_bevfusion:/workspace/autoware/src/autoware_lidar_bevfusion -v $(pwd)/autoware_tensorrt_common:/workspace/autoware/src/autoware_tensorrt_common -it bevfusion_docker
```

# Compile the package

```
source /opt/ros/humble/setup.bash 
cd autoware
colcon build --symlink-install --continue-on-error --packages-up-to autoware_lidar_bevfusion --event-handlers console_direct+ --cmake-args -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=Release
```

# Launch the node

```
source install/setup.bash
ros2 launch autoware_lidar_bevfusion lidar_bevfusion.launch.xml model_path:=/workspace/autoware/src/autoware_lidar_bevfusion/config
```

# Benchmark

We provide some scripts to test and benchmark the inference of BEVFusion

```bash
./build/autoware_lidar_bevfusion/build_camera_lidar_engine \
    /workspace/autoware/src/autoware_lidar_bevfusion/config/bevfusion_traveller59.onnx \
    /workspace/autoware/build/autoware_lidar_bevfusion/libautoware_tensorrt_plugins.so \
    /workspace/autoware/bevfusion_traveller59.engine 0
./build/autoware_lidar_bevfusion/run_camera_lidar_engine \
    build/autoware_lidar_bevfusion/libautoware_tensorrt_plugins.so \
    bevfusion_traveller59.engine 0

./build/autoware_lidar_bevfusion/build_camera_lidar_engine \
    /workspace/autoware/src/autoware_lidar_bevfusion/config/bevfusion_cl_traveller59.onnx \
    /workspace/autoware/build/autoware_lidar_bevfusion/libautoware_tensorrt_plugins.so \
    /workspace/autoware/bevfusion_cl_traveller59.engine 1
./build/autoware_lidar_bevfusion/run_camera_lidar_engine \
    build/autoware_lidar_bevfusion/libautoware_tensorrt_plugins.so \
    bevfusion_cl_traveller59.engine 1
```

| Modailty | GPU     | Backend  | Precision | #Voxels | Time [ms] |
|----------|---------|----------|-----------|---------|-----------|
| L        | RTX3060 | Pytorch  | fp32      | 68k     | 412.3     |
| CL       | RTX3060 | Pytorch  | fp32      | 68k     | 4103.3    |
| L        | RTX3060 | TensorRT | fp32      | 68k     | 56.1      |
| CL       | RTX3060 | TensorRT | fp32      | 68k     | 132.5     |
| L        | RTX3090 | TensorRT | fp32      | 68k     | 25.6      |
| CL       | RTX3090 | TensorRT | fp32      | 68k     | 56.0      |

Memory usage (including preprocessing buffers):
 - L: 1428MB
 - CL: 3072MB


# Notes

 - BEVFusion lidar only and camera-lidar are compatible, although the current code only allows for camera-lidar
 - Can not be integrated into autoware yet due to TensorRT 10
 - The logic to handle missing cameras has not been implemented
 - The logic to trigger detection has not been fully implemented (depending on the sensor setup, the lidar may not trigger the inference)
 - The results should improve a few milliseconds since the onnx has redundant operations
 - If the latency needs to be further reduced, fp16 is needed. The current code does not handle that, but is not difficult