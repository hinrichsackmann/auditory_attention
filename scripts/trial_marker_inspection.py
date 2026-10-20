import h5py
import numpy as np
from src.loading import scalar

# These .mat files are MATLAB 7.3 (HDF5-based), which scipy.io.loadmat can't
# read -- h5py is used instead. Groups are MATLAB structs, Datasets with a
# reference dtype are MATLAB cell arrays (pointing into the shared '#refs#'
# pool), and 'char' datasets are stored as uint16 codepoints.

_REF_DTYPE = h5py.special_dtype(ref=h5py.Reference)


def _decode_if_char(dataset):
    if dataset.attrs.get("MATLAB_class") == b"char":
        return "".join(chr(c) for c in np.asarray(dataset[()]).ravel())
    return None


def inspect_markers(filepath):
    with h5py.File(filepath, "r") as f:
        # Skip the internal HDF5 reference pool
        keys = [k for k in f.keys() if k != "#refs#"]
        print(f"=== Top-level keys: {keys}\n")

        for key in keys:
            obj = f[key]
            print(f"--- Key: '{key}' ---")
            print(f"  Type:  {type(obj)}")

            # Plain dataset (array or cell array)
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")

                if obj.dtype == _REF_DTYPE:
                    print(f"  Cell array with {obj.size} element(s)")
                else:
                    text = _decode_if_char(obj)
                    if text is not None:
                        print(f"  Values: {text!r}")
                    elif obj.ndim <= 2 and obj.size < 200:
                        print(f"  Values: {obj[()]}")

            # MATLAB struct (HDF5 group)
            elif isinstance(obj, h5py.Group):
                fields = list(obj.keys())
                print(f"  Struct fields: {fields}")
                for field in fields:
                    val = obj[field]
                    if isinstance(val, h5py.Dataset):
                        print(f"    .{field}: shape={val.shape}, dtype={val.dtype}")
                        if val.dtype == _REF_DTYPE:
                            print(f"      cell array with {val.size} element(s)")
                        else:
                            text = _decode_if_char(val)
                            if text is not None:
                                print(f"      = {text!r}")
                            elif val.size < 50:
                                print(f"      = {val[()]}")

            print()


#inspect_markers("data/raw/S1.mat")

def inspect_hdf5(filepath):
    with h5py.File(filepath, 'r') as f:
        
        def print_item(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}[Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}[Group]   {name}/")
        
        f.visititems(print_item)

#inspect_hdf5('data/raw/S1.mat')


'''
with h5py.File('data/raw/S1.mat', 'r') as f:
    # Get a reference to the first trial
    trial_ref = f['trials'][0, 0]   # cell array reference
    trial = f[trial_ref]
    
    print("Trial 0 keys:", list(trial.keys()))
    
    # Print shape of each field
    for key in trial.keys():
        item = trial[key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            # Print a small sample if it looks like scalars or labels
            if item.size < 20:
                print(f"    values: {item[()]}")
'''

with h5py.File('data/raw/S1.mat', 'r') as f:
    trial_ref = f['trials'][0, 0]
    trial = f[trial_ref]
    
    # Helper to decode ASCII uint arrays
    def decode_str(arr):
        return ''.join(chr(int(c)) for c in arr.flatten())
    
    # Confirm the label fields
    print("attended_ear:", decode_str(trial['attended_ear'][()]))  # expect 'R' or 'L'
    print("attended_track:", scalar(trial['attended_track'][()]))      # expect 1 or 2
    print("condition:", decode_str(trial['condition'][()]))         # expect 'hrtf' etc.
    print("subject:", decode_str(trial['subject'][()]))             # expect 'S1' etc.
    
    # Explore RawData
    print("\nRawData contents:")
    for key in trial['RawData'].keys():
        item = trial['RawData'][key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
    
    # Explore FileHeader
    print("\nFileHeader contents:")
    for key in trial['FileHeader'].keys():
        item = trial['FileHeader'][key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            if item.size < 5:
                print(f"    = {item[()]}")
    
    # Dereference one stimulus
    print("\nStimulus 0 contents:")
    stim_ref = trial['stimuli'][0, 0]
    stim = f[stim_ref]

    print(f"  Type: {type(stim)}")
    print(f"  Shape: {stim.shape}")
    print(f"  Dtype: {stim.dtype}")

    # If it's a group after all, list keys; if dataset, show a sample
    if isinstance(stim, h5py.Group):
        for key in stim.keys():
            item = stim[key]
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
    elif isinstance(stim, h5py.Dataset):
        print(f"  Direct array — first few values: {stim[0, :5]}")

    print("\nStimulus 1 contents:")
    stim_ref2 = trial['stimuli'][0, 1]
    stim2 = f[stim_ref2]
    print(f"  Type: {type(stim2)}, Shape: {stim2.shape}, Dtype: {stim2.dtype}")

with h5py.File('data/raw/S1.mat', 'r') as f:
    trial_ref = f['trials'][0, 0]
    trial = f[trial_ref]
    
    stim_ref = trial['stimuli'][0, 0]
    stim = f[stim_ref]
    print("Stimulus 0 filename:", decode_str(stim[()]))
    
    stim_ref2 = trial['stimuli'][0, 1]
    stim2 = f[stim_ref2]
    print("Stimulus 1 filename:", decode_str(stim2[()]))

with h5py.File('data/raw/S1.mat', 'r') as f:
    trial_ref = f['trials'][0, 0]
    trial = f[trial_ref]
    
    print("RawData contents:")
    raw = trial['RawData']
    if isinstance(raw, h5py.Group):
        for key in raw.keys():
            item = raw[key]
            if isinstance(item, h5py.Dataset):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"  {key}: [Group]")
    elif isinstance(raw, h5py.Dataset):
        print(f"  Direct dataset: shape={raw.shape}, dtype={raw.dtype}")