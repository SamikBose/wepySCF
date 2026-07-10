from pathlib import Path
import os.path as osp
from wepy.hdf5 import WepyHDF5

base_path = '/path/to/your/base/folder/'
hdf5_filenames = [f'{base_path}/sub_0/wb97x_v_aug-cc-pVDZ.wepy.h5',f'{base_path}/sub_1/wb97x_v_aug-cc-pVDZ.wepy.h5']

init_wepy_h5 = WepyHDF5(hdf5_filenames[0], mode='r')
#init_wepy_h5

with init_wepy_h5:
    wepy_h5 = init_wepy_h5.clone(path=osp.join(base_path, 'clone.h5'), mode='w')

with wepy_h5:
    for hdf5_filename in hdf5_filenames:
        wepy_h5.link_file_runs(hdf5_filename)
with wepy_h5:
    print(wepy_h5.num_runs)
    for i in range(wepy_h5.num_runs):
        print(f'Number of cycles in run{i} is {wepy_h5.num_run_cycles(i)}...')


