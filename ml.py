from thalpy.analysis import fc, pc, masks, denoise, plotting
import thalpy.base as base

import numpy as np
import nilearn
import pandas as pd
from nilearn.decoding import Decoder
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_DIR = "/Shared/lss_kahwang_hpc/data/HCP_D/"
# DATASET_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/HCP_D/"
DIR_TREE = base.DirectoryTree(DATASET_DIR)
INTERVIEW_AGE = "interview_age"


def calc_func(n_masker, m_masker):
    fc_data = fc.FcData(
        DATASET_DIR,
        n_masker,
        m_masker,
        "cortical_fc",
        task="rest",
        cores=80,
    )
    fc_data.calc_fc(stack=True, stack_size=1)

    return fc_data


masker = masks.get_roi_mask(masks.SCHAEFER_YEO7_PATH)
calc_func(masker, masker)
