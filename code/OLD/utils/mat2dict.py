def mat2dict(mat_array):
    '''
    reformat object numpy array returned by scipy.io.loadmat for mat files as dictionaries
    '''
    mat_dict = {field_nm: mat_array[field_nm][:] for field_nm in mat_array.dtype.names}
    return mat_dict
