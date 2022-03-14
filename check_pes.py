import json
import glob

fmap_dir = '/Dedicated/inc_data/HCP_D/rawdata/sub-0001305/fmap/'
func_dir = '/Dedicated/inc_data/HCP_D/rawdata/sub-0001305/func/'

func_files = sorted(glob.glob(func_dir + '*bold.json'))


print('File     PhaseEncodingDirection   IntendedFor')
for file in func_files:
    with open(file) as f:
        func_metadata = json.load(f)
        print(
            f'{file.split("/")[-1]}  {func_metadata["PhaseEncodingDirection"]}')

    fmap_file = file.replace('func', 'fmap').replace(
        'task', 'acq').replace('bold', 'fieldmap')
    with open(fmap_file) as f:
        fmap_metadata = json.load(f)
        print(
            f'{fmap_file.split("/")[-1]}   {fmap_metadata["PhaseEncodingDirection"]}   IntendedFor: {fmap_metadata["IntendedFor"]}')
        print()
