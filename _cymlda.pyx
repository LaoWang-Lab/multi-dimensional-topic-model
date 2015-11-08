#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

def _train_corpus(int[:,:,:] nhet, int[:,:] nhe, int[:,:] nmh, int[:] tmw, int[:] hmw, int[:,:] nmw, int[:,:] zmh, double alpha, double beta, double gamma):
    cdef:
        int m, wc, w, hw
        int M = nmh.shape[0]
        int H = nhe.shape[0]

    # for each document m
    for m in range(M):
        wc = nmw[0][m] # word count
        # sample dimension of each word
        for w in range(wc):
            _sample_dim(nhet, nhe, nmh, tmw, hmw, nmw, zmh, alpha, beta, gamma, m, w)
        # batch sample topic of each word
        for hw in range(H):
            _sample_topic(nhet, nhe, nmh, tmw, hmw, nmw, zmh, alpha, beta, gamma, m, hw)

cdef _sample_dim(int[:,:,:] nhet, int[:,:] nhe, int[:,:] nmh, int[:] tmw, int[:] hmw, int[:,:] nmw,
                 int[:,:] zmh, double alpha, double beta, double gamma, int m, int w):
    cdef:
        int t, h, z
        double cum
        int hw = hmw[nmw[1][m] + w]
        int tw = tmw[nmw[1][m] + w]
        int zw = zmh[m, hw]
        int H = nhe.shape[0]
        int T = nhet.shape[2]
        double[:] theta = np.zeros(H)
        double[:] phi_t = np.zeros(H)
        double[:] tmp = np.zeros(H)

    dec(nmh[m, hw])
    dec(nhet[hw, zw, tw])

    # theta = (self._n_mh[m] + alpha) / (self._n_mh[m] + alpha).sum()
    cum = 0
    for h in range(H):
        theta[h] = nmh[m,h] + alpha
        cum += theta[h]
    for h in range(H):
        theta[h] /= cum

    # phi_t = np.array([get_phi_ht(h) for h in range(H)])
    for h in range(H):
        cum = 0
        z = zmh[m,h]
        for t in range(T):
            cum += nhet[h,z,t] + beta
        phi_t[h] = (nhet[h,z,tw] + beta) / cum

    # tmp = (theta*phi_t)/(theta*phi_t).sum()
    cum = 0
    for h in range(H):
        tmp[h] = theta[h] * phi_t[h]
        cum += tmp[h]
    for h in range(H):
        tmp[h] /= cum

    hw = np.nonzero(np.random.multinomial(1, np.asarray(tmp)))[0][0]
    zw = zmh[m, hw]

    inc(nmh[m, hw])
    inc(nhet[hw, zw, tw])
    hmw[nmw[1][m] + w] = hw

cdef _sample_topic(int[:,:,:] nhet, int[:,:] nhe, int[:,:] nmh, int[:] tmw, int[:] hmw, int[:,:] nmw,
                   int[:,:] zmh, double alpha, double beta, double gamma, int m, int hw):
    cdef:
        int wi, tw, w, e, ee
        double r, cum, tcum, pTi
        int zw = zmh[m, hw]
        int E = nhet.shape[1]
        int W = nmw[0][m]
        int T = nhet.shape[2]
        double[:] tmp = np.zeros(E)
        list wI = list()

    dec(nhe[hw, zw])

    # find all word assined to dimension hw
    for w in range(W):
        if hmw[nmw[1][m] + w] == hw:
            tw = tmw[nmw[1][m] + w]
            # print(m, hw, zw, w, hmw[nmw[1][m] + w], nhet.shape)
            wI.append(w)
            dec(nhet[hw, zw, tw])

    # tmp = np.array([float(pZ(e)) for e in range(E)])
    tcum = 0
    for e in range(E):
        cum = 0
        for ee in range(E):
            cum += nhe[hw, ee] + gamma
        r = (nhe[hw, e] + gamma) / cum
        for wi in wI:
            tw = tmw[nmw[1][m] + wi]
            cum = 0
            for t in range(T):
                cum += nhet[hw, e, t] + beta
            pTi = (nhet[hw, e, tw] + beta) / cum
            r *= pTi

        tmp[e] = r
        tcum += r
    for e in range(E):
        tmp[e] /= tcum
    zw = np.nonzero(np.random.multinomial(1, np.asarray(tmp)))[0][0]

    zmh[m, hw] = zw
    inc(nhe[hw, zw])
    for wi in wI:
        inc(nhet[hw, zw, tmw[nmw[1][m] + wi]])