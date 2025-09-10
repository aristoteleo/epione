# cython: language_level=3

"""
Cython optimized footprint calculations for epione
Adapted from TOBIAS signals.pyx

@author: Zehua Zeng (adapted from Mette Bentsen)
@contact: starlitnightly@gmail.com
@license: GPL 3.0

High-performance Cython implementations of footprint algorithms
"""

import numpy as np
cimport numpy as np
import cython
from cython cimport boundscheck, wraparound, cdivision
from libc.math cimport fabs, INFINITY
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time


ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t INT_t

#--------------------------------------------------------------------------------------------------#
@cython.boundscheck(False)	#dont check boundaries
@cython.cdivision(True)		#no check for zero division
@cython.wraparound(False) 	#dont deal with negative indices
def tobias_footprint_array(np.ndarray[np.float64_t, ndim=1] arr, int flank_min, int flank_max, int fp_min, int fp_max):
    """
    Calculate TOBIAS footprint scores for an array using Cython optimization
    
    Parameters
    ----------
    arr : np.ndarray[float64]
        1D array of signal values
    flank_min : int
        Minimum flank window size
    flank_max : int
        Maximum flank window size
    fp_min : int
        Minimum footprint window size
    fp_max : int
        Maximum footprint window size
        
    Returns
    -------
    np.ndarray[float64]
        Array of footprint scores
    """

    cdef int L = arr.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] footprint_scores = np.zeros(L)
    cdef int i, j, footprint_w, flank_w, pos

    cdef float fp_sum, fp_mean, fp_score, flank_mean
    cdef float left_sum, right_sum, val

    # Each position in array, starting at i, going until last possible start of region
    for i in range(L-2*flank_max-fp_max):
        for flank_w in range(flank_min, flank_max+1):
            for footprint_w in range(fp_min, fp_max+1):

                # Sum of left flank
                left_sum = 0.0
                for j in range(flank_w):
                    val = arr[i+j]
                    if val > 0.0:
                        left_sum += val

                # Sum of footprint (only negative counts)
                fp_sum = 0.0
                for j in range(footprint_w):
                    val = arr[i+flank_w+j]	
                    if val < 0.0:
                        fp_sum += val 	#val is negative

                # Sum of right flank
                right_sum = 0.0
                for j in range(flank_w):
                    val = arr[i+flank_w+footprint_w+j]
                    if val > 0.0:
                        right_sum += val

                # Calculate score
                fp_mean = fp_sum/(1.0*footprint_w)	#will be minus
                flank_mean = (right_sum + left_sum)/(2.0*flank_w)

                fp_score = flank_mean - fp_mean 	#-- = +

                # Save score across footprint
                for pos in range(i+flank_w, i+flank_w+footprint_w):
                    if fp_score > footprint_scores[pos]:
                        footprint_scores[pos] = fp_score

    return footprint_scores


#--------------------------------------------------------------------------------------------------#
@cython.cdivision(True)		#no check for zero division
@cython.boundscheck(False)	#dont check boundaries
@cython.wraparound(False) 	#dont deal with negative indices
def fos_score_array(np.ndarray[np.float64_t, ndim=1] arr, int flank_min, int flank_max, int fp_min, int fp_max):
    """
    Calculate FOS (Footprint Occupancy Score) for an array using Cython optimization
    
    Parameters
    ----------
    arr : np.ndarray[float64]
        1D array of signal values
    flank_min : int
        Minimum flank window size
    flank_max : int
        Maximum flank window size
    fp_min : int
        Minimum footprint window size
    fp_max : int
        Maximum footprint window size
        
    Returns
    -------
    np.ndarray[float64]
        Array of FOS scores
    """
    
    cdef int L = arr.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] footprint_scores = np.zeros(L)
    footprint_scores[:] = 10000
    cdef int i, j, footprint_w, flank_w

    cdef float Lm, Cm, Rm, fos_score
    cdef float left_sum, right_sum, fp_sum

    # Each position in array, starting at i, going until last possible start of region
    for i in range(L-2*flank_max-fp_max):
        for flank_w in range(flank_min, flank_max):
            for footprint_w in range(fp_min, fp_max):
            
                # Sum of left flank
                left_sum = 0.0
                for j in range(flank_w):
                    left_sum += arr[i+j]

                # Sum of footprint
                fp_sum = 0.0
                for j in range(footprint_w):
                    fp_sum += arr[i+flank_w+j]

                # Sum of right flank
                right_sum = 0.0
                for j in range(flank_w):
                    right_sum += arr[i+flank_w+footprint_w+j]

                # Calculate score
                Lm = left_sum / (flank_w*1.0)		#left mean
                Cm = fp_sum / (footprint_w*1.0)	#center mean
                Rm = right_sum / (flank_w*1.0)		#right mean

                if Cm < Rm and Cm < Lm and Lm > 0 and Rm > 0:
                    fos_score = (Cm+1.0)/Lm + (Cm+1.0)/Rm
                else:
                    fos_score = 10000

                # Save score to arr (smallest are best)
                for j in range(footprint_w):
                    if fos_score < footprint_scores[i+flank_w+j]:
                        footprint_scores[i+flank_w+j] = fos_score

    return footprint_scores


#--------------------------------------------------------------------------------------------------#
@cython.boundscheck(False)	#dont check boundaries
@cython.cdivision(True)		#no check for zero division
@cython.wraparound(False) 	#dont deal with negative indices
def fast_rolling_mean(np.ndarray[np.float64_t, ndim=1] arr, int w):
    """
    Fast rolling mean calculation using Cython
    
    Parameters
    ----------
    arr : np.ndarray[float64]
        Input array
    w : int
        Window size
        
    Returns
    -------
    np.ndarray[float64]
        Array of rolling means
    """

    cdef int L = arr.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] roll_arr = np.zeros(L)
    roll_arr[:] = np.nan
    cdef int i, j
    cdef int lf = int(np.floor(w / 2.0))
    cdef float valsum

    # Mean in window
    for i in range(L-w+1):
        valsum = 0
        for j in range(w):
            valsum += arr[i+j]

        roll_arr[i+lf] = valsum/(w*1.0)

    return roll_arr


#--------------------------------------------------------------------------------------------------#
@cython.boundscheck(False)	#dont check boundaries
@cython.cdivision(True)		#no check for zero division
@cython.wraparound(False) 	#dont deal with negative indices
def fast_rolling_max(np.ndarray[np.float64_t, ndim=1] arr, int w):
    """
    Fast rolling maximum calculation using Cython
    
    Parameters
    ----------
    arr : np.ndarray[float64]
        Input array
    w : int
        Window size
        
    Returns
    -------
    np.ndarray[float64]
        Array of rolling maxima
    """

    cdef int L = arr.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] roll_arr = np.zeros(L)
    cdef int i, j, start_i
    cdef int lf = int(np.floor(w / 2.0))
    cdef float maxval

    # Max in window
    for i in range(L):		#centered on reg

        maxval = arr[i]		#initialize with first value
        start_i = i - lf 	#start_i + w gives full region
    
        for j in range(w):
            if start_i + j > 0 and start_i + j < L:
                if arr[start_i+j] > maxval:
                    maxval = arr[start_i+j]
        
        # Assign maxval to index
        roll_arr[i] = maxval

    return roll_arr


#--------------------------------------------------------------------------------------------------#
@cython.boundscheck(False)	#dont check boundaries
@cython.cdivision(True)		#no check for zero division
@cython.wraparound(False) 	#dont deal with negative indices
def fast_rolling_math(np.ndarray[np.float64_t, ndim=1] arr, int w, operation):
    """
    Rolling operation of arr with window size w 
    Possible operations are: "max", "min", "mean", "sum"
    Returns array the same size as arr with "NaN" in flanking positions (for sum/mean)
    """

    cdef int L = arr.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] roll_arr = np.zeros(L)
    roll_arr[:] = np.nan
    cdef int i, j, start_i
    cdef int lf = int(np.floor(w / 2.0))
    cdef int rf = int(np.ceil(w / 2.0)) 
    cdef float minval, maxval, valsum, prod
    
    # Max in window
    if operation == "max":
        for i in range(L):		#centered on reg

            maxval = arr[i]		#initialize with first value
            start_i = i - lf 	#start_i + w gives full region
        
            for j in range(w):
                if start_i + j > 0 and start_i + j < L:
                    if arr[start_i+j] > maxval:
                        maxval = arr[start_i+j]
            
            # Assign maxval to index
            roll_arr[i] = maxval
    
    # Min in window
    elif operation == "min":
        for i in range(L):

            minval = arr[i]
            start_i = i - lf 	

            for j in range(w):
                if start_i + j > 0 and start_i + j < L:
                    if arr[start_i+j] < minval:
                        minval = arr[start_i+j]

            # Assign minval to index
            roll_arr[i] = minval

    # Sum of window
    elif operation == "sum":
        for i in range(L-w+1):
            valsum = 0
            for j in range(w):
                valsum += arr[i+j]

            roll_arr[i+lf] = valsum

    # Mean in window
    elif operation == "mean":
        for i in range(L-w+1):
            valsum = 0
            for j in range(w):
                valsum += arr[i+j]

            roll_arr[i+lf] = valsum/(w*1.0)

    # Product of values in window
    elif operation == "prod":
        for i in range(L):
            prod = 1.0
            start_i = i - lf
            for j in range(w):
                if start_i + j > 0 and start_i + j < L:
                    prod *= arr[start_i+j]

            roll_arr[i] = prod

    return roll_arr


#--------------------------------------------------------------------------------------------------#
def shuffle_array(np.ndarray[np.float64_t, ndim=1] arr, int no_rand, np.ndarray[np.int64_t, ndim=1] shift_options):
    """ Shuffles array of values within the boundaries given in shift """

    cdef int max_shift = max([abs(np.min(shift_options)), abs(np.max(shift_options))])
    cdef np.ndarray[np.float64_t, ndim=1] ext_arr = np.concatenate((np.zeros(max_shift), arr, np.zeros(max_shift)))	 #pad with max shift to allow for shuffling outside borders
    cdef int ext_arr_len = len(ext_arr)

    cdef np.ndarray[np.int64_t, ndim=1] nonzero_index = np.nonzero(ext_arr)[0]
    cdef int no_shift = len(nonzero_index)

    cdef np.ndarray[np.int64_t, ndim=2] rand_rel_positions = np.random.choice(shift_options, size=(no_shift, no_rand)) 		#positions of shuffled reads
    cdef np.ndarray[np.float64_t, ndim=2] rand_mat = np.zeros((no_rand, ext_arr_len))

    cdef double value
    cdef int i, j, pos, rand_pos

    # Get all relative placements
    for i in range(no_shift):

        pos = nonzero_index[i] 	#original index in ext_arr
        value = ext_arr[pos]	#value at original position

        # Get new random positions
        for j in range(no_rand):
            rand_pos = pos + rand_rel_positions[i,j]  #i for cut, j for each randomization
            rand_mat[j, rand_pos] = rand_mat[j, rand_pos] + value

    return rand_mat[:,max_shift:-max_shift]


#--------------------------------------------------------------------------------------------------#
@boundscheck(False)
@wraparound(False)
@cdivision(True)
def quantile_normalize(np.ndarray[DTYPE_t, ndim=2] matrix):
    """
    Perform quantile normalization on matrix columns
    
    Parameters
    ----------
    matrix : np.ndarray[float64, ndim=2]
        Input matrix (samples x features)
        
    Returns
    -------
    np.ndarray[float64, ndim=2]
        Quantile normalized matrix
    """
    
    cdef int i, j, rank
    cdef int n_samples = matrix.shape[0]
    cdef int n_features = matrix.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros((n_samples, n_features), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] sorted_values = np.zeros(n_samples, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] mean_quantiles = np.zeros(n_samples, dtype=np.float64)
    cdef np.ndarray[INT_t, ndim=1] ranks = np.zeros(n_samples, dtype=np.int64)
    
    # Calculate mean quantiles
    for rank in range(n_samples):
        mean_quantiles[rank] = 0.0
        for j in range(n_features):
            # Get sorted values for this feature
            for i in range(n_samples):
                sorted_values[i] = matrix[i, j]
            
            # Sort values (simple insertion sort for small arrays)
            for i in range(1, n_samples):
                key = sorted_values[i]
                j = i - 1
                while j >= 0 and sorted_values[j] > key:
                    sorted_values[j + 1] = sorted_values[j]
                    j -= 1
                sorted_values[j + 1] = key
            
            mean_quantiles[rank] += sorted_values[rank]
        
        mean_quantiles[rank] /= n_features
    
    # Apply quantile normalization
    for j in range(n_features):
        # Get ranks for this feature
        for i in range(n_samples):
            sorted_values[i] = matrix[i, j]
        
        # Calculate ranks (simplified ranking)
        for i in range(n_samples):
            rank = 0
            for k in range(n_samples):
                if matrix[k, j] < matrix[i, j]:
                    rank += 1
            ranks[i] = rank
            
        # Assign normalized values
        for i in range(n_samples):
            result[i, j] = mean_quantiles[ranks[i]]
    
    return result