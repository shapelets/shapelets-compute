#%%

import shapelets.compute as sc 
import math 

ts = sc.array([1.0790, 0.0268, 0.0577, -0.0266, 0.7945, 1.2280, -0.1245, 0.3900, 0.6878, 0.9278, 1.5057, 1.7477, 2.1118, 0.7481, 0.8873, -0.0062, 0.6475, 1.2384, 2.8441, 
2.9297, 2.2216, 1.6092, 0.5585, -0.9340, 0.7967, 3.1336, 2.6121, 2.6310, 3.3677, 2.0166, 2.0820, 2.9492, 1.7101, 1.5500, 4.2419, 4.3408, 4.2208, 4.4731, 4.3159, 
4.4276, 2.5431, 3.2336, 0.4730, 0.1313, -0.4484, 0.0909, -1.0016, 0.0513, -0.9547, -0.1819, -0.4218, -1.0246, -1.7997, -3.6256, -4.0557, -3.6010, -3.3663, 
 -3.2584, -4.1531, -5.2027, -5.5468, -6.3903, -7.5082, -6.5579, -6.6629, -8.3958, -9.9829, -10.1461, -9.0033, -8.5042, -10.7809, -11.0123, -11.0221, 
-11.0183, -9.7231, -10.1347, -8.9281, -7.9136, -6.3657, -6.5432, -7.3858, -7.1259, -8.4765, -8.5431, -8.4998, -8.0623, -6.5513, -7.6980, -10.1845, 
-12.1801, -11.6224, -10.1635, -10.1887, -10.6473, -13.0170, -13.7439, -13.2686, -13.4675, -13.0886, -12.5886])

#%%

def discord_discovery_gemm3(ts: sc.ShapeletsArray, subseqlen: int, guessed_r: float):
    if guessed_r <=0 or guessed_r > (2.0 * math.sqrt(subseqlen)):
        raise ValueError("r should be between 0 and 2.0*sqrt(L)")

    if (math.floor(len(ts) / 2.0) < subseqlen) or subseqlen < 4:
        raise ValueError("Subsequence length must in range 4 < L <= floor(len(ts)/2)")        

    # this could be a parameter 
    min_sep = subseqlen
    
    # subsequent count 
    subseqcount = len(ts) - subseqlen + 1
    # transform znorm euc to correlation coeff.
    r = 1.0 - (guessed_r**2.0) / (2.0 * subseqlen)

    ss = sc.unpack(ts, subseqlen, 1, 1, 1)
    ss = ss - sc.tile(sc.sum(ss, 0) / subseqlen, subseqlen)
    ss = ss / sc.tile(sc.power(sc.sum(sc.power(sc.absolute(ss), 2.0), 0), 0.5), subseqlen)

    mxblocklen = 256
    cands = sc.zeros((subseqlen, subseqcount), ts.dtype)
    cands_idx = sc.zeros(subseqcount, ts.dtype)
    cand_count = 0

    for i in range(0, subseqcount, mxblocklen):
        blocklen = min(mxblocklen, subseqcount - i)
        no_rej = sc.full(blocklen, True, 'bool')
        if cand_count != 0:
            pass 

        if blocklen >= min_sep:
            for pos in range(min_sep, blocklen):
                cr = sc.matmulTN(ss[:, i + pos], ss[:, i: i + pos - min_sep + 1])
                others = cr > r
                if (sc.any(others)): # ~isempty(others)
                    no_rej[pos] = False 
                    no_rej[others] = False
        
        can_add = sc.flatnonzero(no_rej) 
        if not can_add.is_empty:
            can_add += i
            prev_count = cand_count 
            cand_count = cand_count + len(can_add)
            cands[:, prev_count: prev_count + len(can_add)] = ss[:, can_add]
            cands_idx[prev_count: prev_count + len(can_add)] = can_add
    pass

# %%
discord_discovery_gemm3(ts, 10, 3)

# %%
