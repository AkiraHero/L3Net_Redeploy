import open3d as o3d
import numpy as np
import math


class KeypointSelection:
    def __init__(self, threshold, neighborhood_distance, k_neighbor, keypoint_num):
        pcd = o3d.io.read_point_cloud("1.pcd")
        print(pcd)
        point = np.asarray(pcd.points)
        self.point = point
        self.threshold = threshold                          # 由用户传入，被选为候选点的点云密度阈值
        self.neighborhood_distance = neighborhood_distance  # 由用户传入，计算密度时邻域的大小
        self.k_neighbor = k_neighbor                        # 由用户传入，计算3D structure tensor时k近邻的k值
        self.keypoint_num = keypoint_num                    # 由用户传入，最终综合linearity和scattering以及distance选择的点的个数

    def traverse_density(self):
        pcd = o3d.io.read_point_cloud("1.pcd")  # 读取点云数据
        print(pcd)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])    # 统一着色为灰色
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)    # 构建一个KDTree类
        pointNum = self.point.shape[0]
        selected_index = []                         # 用来存储邻域密度满足要求的点的索引

        for i in range(pointNum):
            if i % 50000 == 0:
                print("候选点选择，进度" + str(round(i/50561.57,3)) + "%")
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], self.neighborhood_distance)  # R半径近邻
            if k >= self.threshold:                                   # 判断第j个点与第i个点距离是否小于阈值
                selected_index.append(i)

        print(selected_index)
        np.asarray(pcd.colors)[selected_index[1:], :] = [0, 1, 0]     # 寻址，将候选点涂成绿色

        return pcd, pcd_tree, selected_index

    def structure_tensor(self):
        pcd, pcd_tree, selected_index = self.traverse_density()       # 调用类中的第一个函数
        structure_tensor = np.zeros((len(selected_index), 3, 3))
        print('共选择到' + str(len(selected_index)) + '个候选点')

        for i in range(len(selected_index)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], self.k_neighbor)  # k近邻
            neighbor_point = pcd.select_down_sample(idx)
            neighbor_array = np.asarray(neighbor_point.points)
            mean = k * neighbor_array.mean(axis=0) / (k+1)            # 选择的k个邻点的均值坐标——1*3
            if i % 500 == 0:
                print("structure_tensor计算，第 "+str(i) + "/" + str(len(selected_index)) + " 个候选点")

            S = np.zeros((3, 3))                                      # 用来存放当前候选点的3D structure tensor
            for j in range(k):
                temp_point = neighbor_array[j] - mean
                temp_point1 = temp_point.reshape(3, 1)                # 为了进行列向量乘以行向量，要对其进行转置
                temp_point2 = temp_point1.reshape(1, 3)
                S = S + temp_point1.dot(temp_point2)
            structure_tensor[i] = S / (k + 1)

        return pcd, selected_index, structure_tensor

    def combinatorially_geometric_characteristic(self):
        scatter_w = 1  # scatter相较于linear的权重
        pcd, selected_index, structure_tensor = self.structure_tensor()            # 调用类中的第二个函数
        selected_point = pcd.select_down_sample(selected_index)
        number_of_point = structure_tensor.shape[0]
        combined_geometric_characteristic = np.zeros(number_of_point)              # 每个点组合几何特征
        geometrical_feature = np.zeros((number_of_point, 3))                       # 第i行是候选点中第i个点的linear, planar, scatter

        for i in range(number_of_point):
            print("组合几何特征计算，第" + str(i) + "个候选点")
            e_vals, e_vecs = np.linalg.eig(structure_tensor[i])                    # 求第i个点的3D structure tensor的特征向量和特征值
            e_vals_sort = np.sort(e_vals)[::-1]                                    # 将特征值从大到小排列
            sigma_1 = math.sqrt(e_vals_sort[0])
            sigma_2 = math.sqrt(e_vals_sort[1])
            sigma_3 = math.sqrt(e_vals_sort[2])
            geometrical_feature[i][0] = (sigma_1 - sigma_2) / sigma_1              # 邻域的linear
            geometrical_feature[i][1] = (sigma_2 - sigma_3) / sigma_1              # 邻域的planar
            geometrical_feature[i][2] = (sigma_3 / sigma_1)                        # 邻域的scatter
            combined_geometric_characteristic[i] = geometrical_feature[i][0] + scatter_w * geometrical_feature[i][2]

        temp_index = np.argsort(combined_geometric_characteristic)
        temp_index = temp_index[0:self.keypoint_num]
        keypoints = selected_point.select_down_sample(temp_index)

        keypoint_index = []
        for i in range(len(temp_index)):
            keypoint_index.append(selected_index[temp_index[i]])

        np.asarray(pcd.colors)[keypoint_index[1:], :] = [1, 0, 0]     # 寻址，将关键点涂成绿色
        o3d.visualization.draw_geometries([pcd])

        return keypoints
