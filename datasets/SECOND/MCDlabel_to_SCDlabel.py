import os

import numpy as np
from skimage import io

MCD_numcls = 7
MCD_COLORMAP = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]
MCD_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

SCD_COLORMAP = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]
SCD_ClASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
MAP_A = [0, 1, 2, 3, 3, 4, 4]
MAP_B = [0, 1, 2, 3, 3, 4, 4]


def is_img(ext):
    ext = ext.lower()
    if ext == '.jpg':
        return True
    elif ext == '.png':
        return True
    elif ext == '.jpeg':
        return True
    elif ext == '.bmp':
        return True
    elif ext == '.tif':
        return True
    else:
        return False


colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(SCD_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        data = data.astype(np.int32)
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        IndexLabels.append(colormap2label[idx])
    return IndexLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return colormap2label[idx]


def Index2Color(mask):
    colormap = np.asarray(SCD_COLORMAP, dtype='uint8')
    x = np.asarray(mask, dtype='int32')
    return colormap[x, :]


# def mapping(MCD_idx):
#     funcA = mappingA[MCD_idx]
#     SCD_idxB = mappingB[MCD_idx]
#     return SCD_idxA, SCD_idxB

def MCD2SCD(label):
    # h, w = Img.shape
    # funcA = lambda idx: mappingA[idx]
    # funcB = lambda idx: mappingB[idx]
    # labelA = funcA(label)
    # labelB = funcB(label)
    label = np.asarray(label)
    mapA = np.asarray(MAP_A)
    mapB = np.asarray(MAP_B)
    labelA = mapA[label]
    labelB = mapB[label]
    return labelA.astype(np.uint8), labelB.astype(np.uint8)


def main():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    SrcDir = os.path.join(working_dir, 'DATA_ROOT\\label2_rgb')
    # DstDirA = os.path.join(working_dir, 'DATA_ROOT\\label')
    DstDirB = os.path.join(working_dir, './DATA_ROOT/label2')
    # if not os.path.exists(DstDirA): os.makedirs(DstDirA)
    if not os.path.exists(DstDirB): os.makedirs(DstDirB)

    data_list = os.listdir(SrcDir)
    for idx, it in enumerate(data_list):
        if (it[-4:] == '.png'):
            src_path = os.path.join(SrcDir, it)
            # dst_pathA = os.path.join(DstDirA, it)
            dst_pathB = os.path.join(DstDirB, it)
            label = io.imread(src_path)
            # labelA, labelB = Color2Index(label)
            # labelA = Color2Index(label).astype(np.uint8)
            labelB = Color2Index(label).astype(np.uint8)
            # io.imsave(dst_pathA, labelA, check_contrast=False)
            io.imsave(dst_pathB, labelB, check_contrast=False)
            if not idx % 50: print('%d/%d labels processed.' % (idx, len(data_list)))
    print('Done.')


if __name__ == '__main__':
    main()
