import numpy as np

def save(path, parameters_list):
    array_dict = {f"p{i}": arr.data for i, arr in enumerate(parameters_list)}
    np.savez(path,**array_dict)

def load(path, parameters_list):
    """
    Mutates the parameters_list in place without changing the reference of each numpy array inside the list.
    """
    data = np.load(path)
    if len(data.files) != len(parameters_list):
        raise ValueError(f"checkpoint has {len(data.files)} tensors but model has {len(parameters_list)} — architecture mismatch")
    for i in range(len(parameters_list)):
        parameters_list[i].data[...] = data[f"p{i}"]

    
