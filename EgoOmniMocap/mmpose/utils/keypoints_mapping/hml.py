#  Copyright Jian Wang @ MPI-INF (c) 2023.
hml_joint_names = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]


def values_term(i):
    i -= 1
    return [4 + i * 3, 4 + i * 3 + 1, 4 + i * 3 + 2] + [4 + 63 + i * 6 + k for k in range(6)] + [
        4 + 63 + 126 + (i + 1) * 3 + k for k in range(3)]


class PartSeg:
    def __init__(self):
        self.part_names = [
            'root',
            'head',
            'left_hand',
            'right_hand',
            'left_foot',
            'right_foot',
        ]
        self.partSeg = [[0, 1, 2, 3, 4 + 63 + 126, 4 + 63 + 126 + 1, 4 + 63 + 126 + 2],
                        [x for i in [3, 6, 9, 12, 15] for x in values_term(i)],
                        [x for i in [13, 16, 18, 20] for x in values_term(i)],
                        [x for i in [14, 17, 19, 21] for x in values_term(i)],
                        [x for i in [1, 4, 7, 10] for x in values_term(i)] + [259, 260],
                        [x for i in [2, 5, 8, 11] for x in values_term(i)] + [261, 262]]

    def get_part_seg(self):
        return self.partSeg

