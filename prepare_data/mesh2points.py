import open3d as o3d
import numpy as np
import os


downsampling_points_count_list = [10000, 20000, 40000, 80000]

data_path = '/home/gpuadmin/IJS/Color_PUGeoNet/dataset/test/'
default_save_path = '/home/gpuadmin/IJS/Color_PUGeoNet/other_num/'
mesh_list = os.listdir(data_path)

for downsampling_points_count in downsampling_points_count_list:
    save_path = default_save_path + f'test_{downsampling_points_count}'
    for mesh_name in mesh_list:
        mesh_path = data_path + mesh_name
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()

        # pcd = mesh.sample_points_uniformly(number_of_points=downsampling_points_count)
        pcd = mesh.sample_points_poisson_disk(number_of_points=downsampling_points_count)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        o3d.io.write_point_cloud(save_path + f"/{mesh_name[:-4]}.ply", pcd)
        print(save_path)
        print(f"{mesh_name} downsampling to {downsampling_points_count} operation is completed!!")

