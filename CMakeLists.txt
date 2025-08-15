cmake_minimum_required(VERSION 3.22.1)
project(gl-slam)

set(CMAKE_CXX_STANDARD 17)

#set(OpenCV_DIR "/third_party/opencv/install/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(Pangolin_DIR ${CMAKE_SOURCE_DIR}/third_party/Pangolin/install/lib/cmake/Pangolin)
find_package(Pangolin REQUIRED)
set(g2o_DIR ${CMAKE_SOURCE_DIR}/third_party/g2o/install/lib/cmake/g2o)
find_package(g2o REQUIRED)


include_directories(
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${g2o_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

link_directories(
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    ${OpenCV_LIBRARY_DIRS}
    ${g2o_LIBRARY_DIRS}
    ${Pangolin_LIBRARY_DIRS}
    
)

# Ensure rpath is used during build and runtime
set(CMAKE_SKIP_BUILD_RPATH FALSE)          # Include rpath in build
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)   # Use rpath immediately
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # Add linked library paths

# # OpenCV from third_party
# include_directories(
#     /third_party/opencv/include
# 	/third_party/opencv/install/include
# 	/third_party/opencv/install/include/opencv4
# )
# link_directories(
# 	/third_party/opencv/install/lib
# )

#library - Add you custom cpp code here and link libraries after this 
add_library(core 
    src/core/TensorRTEngineBuilder.cpp    
    src/core/tensorrt_inference.cpp
    src/core/slam_core.cpp
    src/core/lightglue.cpp
    src/core/superpoint.cpp
)
target_include_directories(core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${g2o_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(core
    nvinfer
    nvonnxparser
    cudart
    dl
    libopencv_core.so
    libopencv_imgcodecs.so
    libopencv_imgproc.so
    libopencv_highgui.so
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${EIGEN3_LIBS}
    g2o::core g2o::stuff g2o::types_sba g2o::solver_eigen g2o::opengl_helper
)


add_library(visualization
    src/visualization/visualization.cpp
)
target_include_directories(visualization PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)
target_link_libraries(visualization
    libopencv_core.so
    libopencv_imgcodecs.so
    libopencv_imgproc.so
    libopencv_highgui.so
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
)


add_executable(lightglue_infer src/main.cpp)

target_link_libraries(lightglue_infer
    core
    visualization
)

# # Set rpath for runtime linking
# set_target_properties(lightglue_infer PROPERTIES
#     BUILD_WITH_INSTALL_RPATH TRUE
#     INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/g2o/install/lib"
# )

# # Optional: Print rpath for debugging
# add_custom_command(TARGET lightglue_infer POST_BUILD
#     COMMAND chrpath -l ./lightglue_infer
#     COMMENT "Checking RPATH of lightglue_infer"
# )
