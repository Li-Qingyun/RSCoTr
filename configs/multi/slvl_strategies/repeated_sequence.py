_base_ = '../MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py'

strategy = dict(
    type='repeated_sequence',  # or round_robin
    sequence=[1, 2, 2, 0, 0, 0]
)