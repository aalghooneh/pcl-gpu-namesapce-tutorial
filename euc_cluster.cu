#include <iostream>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/gpu/segmentation/gpu_extract_clusters.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <cuda.h>




int main(int argc, char** argv){
    std::cout<<"..........comparing gpu and cpu clustering..........."<<std::endl;

    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    reader.read("../table_scene_lms400.pcd", *in_cloud);
    
    std::cout<<"\n\n\n\tPoint cloud read with size: "<<in_cloud->size()<<"\n\n"<<std::endl;


    // filter with voxelgrid approach
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(in_cloud);
    vg.setLeafSize(0.01f,0.01f,0.01f);

    auto start_filter=std::chrono::steady_clock::now();
    vg.filter(*out_cloud);
    auto end_filter=std::chrono::steady_clock::now();


    std::cout<<"\t\t\tthe new cloud size is "<<out_cloud->size()<<std::endl;
    std::cout<<"\t\t\ttime took for filtering was: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(end_filter-start_filter).count()<<std::endl;
    std::cout<<"\t\t\tThe 400k points are fed to cpu"<<std::endl;

    // setting the KDtree and cluster in cpu
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree->setInputCloud(in_cloud);
    std::vector<pcl::PointIndices> cluster_indices;

    boost::shared_ptr<pcl::EuclideanClusterExtraction<pcl::PointXYZ>> ec(new pcl::EuclideanClusterExtraction<pcl::PointXYZ>);

    ec->setClusterTolerance(0.02);
    ec->setMinClusterSize(100);
    ec->setMaxClusterSize(25000);
    ec->setSearchMethod(kdtree);
    ec->setInputCloud(in_cloud);
    
    
    auto start_clustering_cpu=std::chrono::steady_clock::now();
    ec->extract(cluster_indices);
    auto end_clustering_cpu=std::chrono::steady_clock::now();

    std::cout<<"\n\n\tThe time took for the cpu version was: "<<
    std::chrono::duration_cast<std::chrono::milliseconds>(end_clustering_cpu-start_clustering_cpu).count()<<std::endl;
    std::cout<<"\tThe size of the cluster indices is: "<<cluster_indices.size()<<std::endl;

    // now for the gpu version
    pcl::gpu::Octree::Ptr octree_device(new pcl::gpu::Octree);
    
    pcl::gpu::Octree::PointCloud gpu_pointcloud;
    pcl::gpu::EuclideanClusterExtraction< pcl::PointXYZ>::PointCloudHostPtr host_in_cloud = in_cloud;


    gpu_pointcloud.upload(in_cloud->points);

    std::cout<<"\n\n\n\tsize of the gpu point cloud: "<<gpu_pointcloud.size()<<std::endl;
    octree_device->setCloud(gpu_pointcloud);
    pcl::gpu::EuclideanClusterExtraction<pcl::PointXYZ> gpu_ec;
    std::vector<pcl::PointIndices> pci;
    
    
    gpu_ec.setClusterTolerance(0.02);
    gpu_ec.setMinClusterSize(100);
    gpu_ec.setMaxClusterSize(25000);
    gpu_ec.setSearchMethod(octree_device);
    gpu_ec.setHostCloud(in_cloud);  
    
    
    auto gpu_start=std::chrono::steady_clock::now();  
    gpu_ec.extract(pci);
    auto gpu_end=std::chrono::steady_clock::now();




    
    std::cout<<"\tThe time took for gpu version was: "<<
    std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count()<<std::endl;
    std::cout<<"\tThe size of cluster inidices is: "<<pci.size()<<" number of clusters"<<std::endl;



    std::cout<<"\n\n\n\tExiting the program"<<std::endl;



    return 0;
}