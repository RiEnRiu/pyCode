#获取当前目录所有源文件，存在DIR_SRCS变量中
AUX_SOURCE_DIRECTORY(. DIR_SRCS_poisson_blending)

#编译为静态库
ADD_LIBRARY (poisson_blending ${DIR_SRCS_poisson_blending})

#设置链接
#TARGET_LINK_LIBRARIES(poisson_blending ${LIB_OPENCV_CORE})
#TARGET_LINK_LIBRARIES(poisson_blending ${LIB_OPENCV_IMGPROC})
#TARGET_LINK_LIBRARIES(poisson_blending ${LIB_OPENCV_IMGCODECS})
#TARGET_LINK_LIBRARIES(poisson_blending ${LIB_OPENCV_HIGHGUI})
TARGET_LINK_LIBRARIES(poisson_blending stl_expand)
TARGET_LINK_LIBRARIES(poisson_blending mask_maker)
