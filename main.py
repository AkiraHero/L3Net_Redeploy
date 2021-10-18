import Keypoint_Selection_KD


if __name__ == '__main__':
    KeypointSelection = Keypoint_Selection_KD.KeypointSelection(1500, 1, 100, 20)
    keypoints = KeypointSelection.combinatorially_geometric_characteristic()
    print('运行完毕')
    print(keypoints)