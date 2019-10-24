#要求cmake版本
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

#项目名称
PROJECT(imAugmentor) 

#lib名称
#set(LIB_BOOST_FILESYSTEM libboost_filesys0tem.a)
#set(LIB_BOOST_SYSTEM libboost_system.a)
#set(LIB_BOOST_PROGRAM_OPTIONS libboost_program_options.a)
#set(LIB_OPENCV_CORE libopencv_core.so)
#set(LIB_OPENCV_IMGPROC libopencv_imgproc.so)
#set(LIB_OPENCV_IMGCODECS libopencv_imgcodecs.so)
#set(LIB_OPENCV_HIGHGUI libopencv_highgui.so)

set(LIB_BOOST_FILESYSTEM libboost_filesystem-vc140-mt-1_63.lib)
set(LIB_BOOST_SYSTEM libboost_system-vc140-mt-1_63.lib)
set(LIB_BOOST_PROGRAM_OPTIONS libboost_program_options-vc140-mt-1_63.lib)
set(LIB_OPENCV_CORE opencv_world340.lib)
set(LIB_OPENCV_IMGPROC opencv_world340.lib)
set(LIB_OPENCV_IMGCODECS opencv_world340.lib)
set(LIB_OPENCV_HIGHGUI opencv_world340.lib)


#设定工程的包含目录
#INCLUDE_DIRECTORIES(. /usr/local/include)
INCLUDE_DIRECTORIES(. D:\\project\\boost_1_63_0 D:\\project\\opencv-3.4\\build\\include)


#设定依赖库目录
#LINK_DIRECTORIES(/usr/local/lib)
LINK_DIRECTORIES(D:\\project\\boost_1_63_0\\lib64-msvc-14.0 D:\\project\\opencv-3.4\\build\\x64\\vc14\\lib)


#设定依赖库
LINK_LIBRARIES(${LIB_BOOST_FILESYSTEM} ${LIB_BOOST_PROGRAM_OPTIONS} ${LIB_OPENCV_CORE} ${LIB_OPENCV_IMGPROC} ${LIB_OPENCV_HIGHGUI} ${LIB_OPENCV_IMGCODECS} ${LIB_BOOST_SYSTEM})

#add_definitions(-std=c++11)
#add_definitions(-O2)

#设定子模块目录
ADD_SUBDIRECTORY(stl_expand)
ADD_SUBDIRECTORY(aug_solver)
ADD_SUBDIRECTORY(poisson_blending)



#设置生成的目标
ADD_EXECUTABLE(aug_helper aug_helper.cpp)
ADD_EXECUTABLE(augmentor augmentor.cpp)
ADD_EXECUTABLE(image_mixer image_mixer.cpp)
ADD_EXECUTABLE(view_png view_png.cpp)

#设置链接
TARGET_LINK_LIBRARIES(aug_helper aug_solver)

TARGET_LINK_LIBRARIES(augmentor aug_solver)
TARGET_LINK_LIBRARIES(augmentor stl_expand)

TARGET_LINK_LIBRARIES(image_mixer poisson_blending)
TARGET_LINK_LIBRARIES(image_mixer stl_expand)
#TARGET_LINK_LIBRARIES(image_mixer pthread)

TARGET_LINK_LIBRARIES(view_png mask_maker)
TARGET_LINK_LIBRARIES(view_png stl_expand)



