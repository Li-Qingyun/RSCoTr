_base_ = '../MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py'

strategy = dict(
    type='weighted_random',
    p=[1, 1, 1]
)