import test_own_method
import Keypoint_Selection_KD


if __name__ == '__main__':
    KeypointSelection = Keypoint_Selection_KD.KeypointSelection(3, 0.1, 20, 90)
    # KeypointSelection = Keypoint_Selection_KD.KeypointSelection(3, 0.1, 20, 100)
    keypoints = KeypointSelection.combinatorially_geometric_characteristic()
    print('运行完毕')
    print(keypoints)
