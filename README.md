
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
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --continue-on-error --packages-up-to autoware_lidar_bevfusion --event-handlers console_direct+ --cmake-args -DCMAKE_VERBOSE_MAKEFILE=ON
```

# Launch the node

```
source install/setup.bash
ros2 launch autoware_lidar_bevfusion lidar_bevfusion.launch.xml model_path:=/workspace/autoware/src/autoware_lidar_bevfusion/config
```

# Notes

 - BEVFusion lidar only and camera-lidar are compatible, although the current code only allows for camera-lidar
 - Can not be integrated into autoware due to TensorRT 10
 - Since it uses mmcv's ops, it requires LibTorch. I will replace it by libspconv in the future
 - It is slow, which is due to MMCV's implementation, some TensorRT concerns and fp32