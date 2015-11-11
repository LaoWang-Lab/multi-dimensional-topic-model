# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free, srand, rand, RAND_MAX
import numpy as np
cimport numpy as np

#cdef extern from "time.h" nogil:
#    ctypedef int time_t
#    time_t time(time_t*)

cpdef _train_corpus(int[:,:,:] nhet, int[:,:] nhe, int[:,:] nmh, int[:] tmw, int[:] hmw, int[:,:] nmw, int[:,:] zmh, double alpha, double beta, double gamma):
#    srand(time(NULL))
    cdef:
        int m, wc, w, hw
        int M = nmh.shape[0]
        int H = nhe.shape[0]
        int E = nhe.shape[1]
        double *theta = <double *>malloc(H*sizeof(double))
        double *phi_t = <double *>malloc(H*sizeof(double))
        double *tmp_h = <double *>malloc(H*sizeof(double))
        double *tmp_e = <double *>malloc(E*sizeof(double))

    for m in range(M):
        wc = nmw[0, m] # word count
        for w in range(wc):
            _sample_dim(nhet, nhe, nmh, tmw, hmw, nmw, zmh, alpha, beta, m, w, theta, phi_t, tmp_h)
        for hw in range(H):
            _sample_topic(nhet, nhe, nmh, tmw, hmw, nmw, zmh, beta, gamma, m, hw, tmp_e)
            
    free(theta)
    free(phi_t)
    free(tmp_h)
    free(tmp_e)

cdef _sample_dim(int[:,:,:] nhet, int[:,:] nhe, int[:,:] nmh, int[:] tmw, int[:] hmw, int[:,:] nmw, int[:,:] zmh, double alpha, double beta, int m, int w, double *theta, double *phi_t, double *tmp_h):
    cdef:
        int t, h, z
        double cum
        int hw = hmw[nmw[1, m] + w]
        int tw = tmw[nmw[1, m] + w]
        int zw = zmh[m, hw]
        int H = nhe.shape[0]
        int T = nhet.shape[2]

    dec(nmh[m, hw])
    dec(nhet[hw, zw, tw])

    cum = 0
    for h in range(H):
        theta[h] = nmh[m,h] + alpha
        cum += theta[h]
    for h in range(H):
        theta[h] /= cum

    for h in range(H):
        cum = 0
        z = zmh[m,h]
        for t in range(T):
            cum += nhet[h,z,t] + beta
        phi_t[h] = (nhet[h,z,tw] + beta) / cum

    cum = 0
    for h in range(H):
        tmp_h[h] = theta[h] * phi_t[h]
        cum += tmp_h[h]
    for h in range(H):
        tmp_h[h] /= cum

    cum = <double>rand() / <double>RAND_MAX
    hw = H - 1
    for h in range(H):
        cum -= tmp_h[h]
        if cum < 0:
            hw = h
            break
        
    zw = zmh[m, hw]

    inc(nmh[m, hw])
    inc(nhet[hw, zw, tw])
    hmw[nmw[1, m] + w] = hw


cdef _sample_topic(int[:,:,:] nhet, int[:,:] nhe, int[:,:] nmh, int[:] tmw, int[:] hmw, int[:,:] nmw, int[:,:] zmh, double beta, double gamma, int m, int hw, double *tmp_e):
    cdef:
        int wi, tw, w, e, ee
        double r, cum, tcum, pTi
        int zw = zmh[m, hw]
        int H = nhet.shape[0]
        int E = nhet.shape[1]
        int T = nhet.shape[2]
        int W = nmw[0, m]
        list wI = list()

    dec(nhe[hw, zw])

    for w in range(W):
        if hmw[nmw[1, m] + w] == hw:
            tw = tmw[nmw[1, m] + w]
            wI.append(w)
            dec(nhet[hw, zw, tw])

    tcum = 0
    for e in range(E):
        cum = 0
        for ee in range(E):
            cum += nhe[hw, ee] + gamma
        r = (nhe[hw, e] + gamma) / cum
        for wi in wI:
            tw = tmw[nmw[1, m] + wi]
            cum = 0
            for t in range(T):
                cum += nhet[hw, e, t] + beta
            pTi = (nhet[hw, e, tw] + beta) / cum
            r *= pTi

        tmp_e[e] = r
        tcum += r
    for e in range(E):
        tmp_e[e] /= tcum

    cum = <double>rand() / <double>RAND_MAX
    zw = E - 1
    for e in range(E):
        cum -= tmp_e[e]
        if cum < 0:
            zw = e
            break

    zmh[m, hw] = zw
    inc(nhe[hw, zw])
    for wi in wI:
        inc(nhet[hw, zw, tmw[nmw[1, m] + wi]])

