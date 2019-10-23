cmake_minimum_required(VERSION 2.8)

project(multiply_engineer)

#添加头文件
#include_directories("./lib/mnn/include/include")
#include_directories("./lib/mnn/include/tools")
#include_directories("./lib/mnn/include/source/core")
#include_directories("./lib/mnn/include/tools/cpp")
#include_directories("./lib/mnn/include/source")
#include_directories("./lib/mnn/include/source/backend")
#include_directories("./lib/mnn/include/source/cv")
#include_directories("./lib/mnn/include/source/math")
#include_directories("./lib/mnn/include/source/shape")
#include_directories("./include")
include_directories("./includes")
include_directories("./includes/3rd_party/imageHelper")
include_directories("./includes/MNN")
#库目录文件
# LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/../lib)  
#LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/../lib/mnn/libs)
#LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/../lib/opencv_libs)
LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/../libs)

set(CMAKE_CXX_STANDARD 11)
##工程文件目录文件
set(hair_SRCS
./includes/MNN/ImageProcess.hpp
./src/Ultra.cpp
)

add_executable(MNN_FaceDetect ${hair_SRCS} )

target_link_libraries(MNN_FaceDetect
                        -lMNN
                        #-lMNN_Vulkan
                        #-lMNN_CL
                        #-lopencv_core
                        #-lopencv_imgcodecs
			#-lopencv_contrib
                       # -lopencv_imgproc
                        #-lopencv_video
			#-lopencv_videostab
                        #-lopencv_videoio

                        )
                     
