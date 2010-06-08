"""Implements the fast hadamard transform"""
all = ["fht1", "fht2"]
import numpy as np
import _C_fht

def fht(arr, **kargs):
    """Fast Hadamard transform
    
    Notes
    -----
    Calls fht1, fht2 or fht3 function according to the dimension of arr
    
    See also
    --------
    fht1, fht2, fht3: specialized function depending of the dimension of arr
    """
    if arr.ndim == 1:
        return fht1(arr)
    elif arr.ndim == 2:
        return fht2(arr, **kargs)
    elif arr.ndim == 3:
        return fht3(arr, **kargs)
    else:
        raise NotImplemented('fht not implemented for dimension > 3')

def fht1(arr):
    """1-dimensional fast hadamard transform
    Input
    -----
    arr: 1-dimensional array
    
    Output
    ------
    oarr : 1-dimensional array of the size of arr

    Notes
    -----
    It is normalized so applying twice is equivalent to identity
    """
    if arr.ndim != 1:
        raise ValueError('Expected 1-dimensional array')
    if not is_power_of_two(arr.shape[0]):
        raise ValueError('array shape must be a power of two')
    oarr = np.empty(arr.shape)
    if arr.dtype is not np.float64:
        iarr = arr.astype(np.float64)
    else:
        iarr = arr
    _C_fht.fht1(iarr, oarr)
    return oarr / np.sqrt(arr.size)

def fht2(arr, axes=(0, 1)):
    """2-dimensional fast hadamard transform

    Input
    -----
    arr: 2-dimensional array
    axes : (int or tuplue) axes along which is applied the 1d transform
    
    Output
    ------
    oarr : 2-dimensional array of the size of arr


    Notes
    -----
    It is normalized so applying twice is equivalent to identity
    """
    if axes == (0, 1) or axes == (1, 0):
        return fht1(arr.flatten()).reshape(arr.shape)
    if not (axes == 0 or axes == 1):
        raise ValueError('axes is either (0, 1) or 0 or 1')
    if arr.ndim != 2:
        raise ValueError('Expected 2-dimensional array')
    if not is_power_of_two(arr.shape[axes]):
        raise ValueError('array shape along axes must be a power of two')
    if arr.dtype is not np.float64:
        iarr = arr.astype(np.float64)
    else:
        iarr = arr
    iarr = iarr.swapaxes(1, axes)
    oarr = np.empty(iarr.shape)
    _C_fht.fht2(iarr, oarr)
    return (oarr / np.sqrt(arr.shape[axes])).swapaxes(1, axes)

def fht3(arr, axes=(0, 1, 2)):
    """3-dimensional fast hadamard transform

    Input
    -----
    arr: 3-dimensional array
    axes : (int or tuplue) axes along which is applied the 1d transform
    
    Output
    ------
    oarr : 3-dimensional array of the size of arr


    Notes
    -----
    It is normalized so applying twice is equivalent to identity
    """
    if np.all(np.sort(axes) == (0, 1, 2)):
        return fht1(arr.flatten()).reshape(arr.shape)
    elif np.all(np.sort(axes) == (0, 1)):
        shape = (np.prod(arr.shape[0:2]), arr.shape[2])
        axes = 0
    elif np.all(np.sort(axes) == (1, 2)):
        shape = (arr.shape[0], np.prod(arr.shape[1:]))
        axes = 1
    elif np.all(np.sort(axes) == (0, 2)):
        iarr = arr.swapaxes(1, 2)
        oarr = fht3(iarr, axes = (0, 1))
        return oarr.swapaxes(1, 2)
    elif axes == 0:
        shape = (arr.shape[0], np.prod(arr.shape[1:]))
        axes = 0
    elif axes == 2:
        shape = (np.prod(arr.shape[0:2]), arr.shape[2])
        axes = 1
    elif axes == 1:
        iarr = arr.swapaxes(1, 2)
        oarr = fht3(iarr, axes = 2)
        return oarr.swapaxes(1, 2)
    return fht2(arr.reshape(shape), axes=axes).reshape(arr.shape)

def is_power_of_two(input_integer):
    """Test if an integer is a power of two"""
    log_int = np.log2(input_integer)
    return (int(log_int) == log_int)
