cmake_minimum_required(VERSION 3.8)
project(euclidean_clustering LANGUAGES CUDA CXX)



find_package(PCL 1.13 REQUIRED)


link_directories(${PCL_LIBRARY_DIRS})
include_directories(${PCL_INCLUDE_DIRS})


add_executable(cluster_extraction euc_cluster.cu)
target_link_libraries(cluster_extraction ${PCL_LIBRARIES})

