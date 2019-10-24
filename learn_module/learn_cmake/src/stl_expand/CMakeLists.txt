#获取当前目录所有源文件，存在DIR_SRCS变量中
AUX_SOURCE_DIRECTORY(. DIR_SRCS_stl_expand)

#编译为静态库
ADD_LIBRARY (stl_expand ${DIR_SRCS_stl_expand})

#设置链接
#TARGET_LINK_LIBRARIES(stl_expand ${LIB_BOOST_FILESYSTEM})
