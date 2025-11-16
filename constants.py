#!/usr/bin/env python3
from enum import IntEnum

#-------------------------- Output Analysis ------------------------------ #

class Stats(IntEnum):
    PATH = 0
    RMSD = 1
    RF_RMSD = 2
    DES_PAE = 3
    DES_PDE = 4
    DES_PLDDT = 5
    FULL_PAE = 6
    FULL_PDE  = 7
    FULL_PLDDT = 8
    FIX_TOTAL_VOL = 9
    FIX_CAVITYAVG = 10
    FIX_CAVITYCOUNT = 11
    DES_TOTAL_VOL = 12
    DES_CAVITYAVG = 13
    DES_CAVITYCOUNT = 14
    LENGTH = 15

#-------------------------- Amino Acids ------------------------------ #
# ProteinMPNN's default alphabet order:
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"  # 20 aa
class AminoAcid(IntEnum):
    A = 0
    C = 1
    D = 2
    E = 3
    F = 4
    G = 5
    H = 6
    I = 7
    K = 8
    L = 9
    M = 10
    N = 11
    P = 12
    Q = 13
    R = 14
    S = 15
    T = 16
    V = 17
    W = 18
    Y = 19

N_SURF_WEIGHT = 1
X_SURF_WEIGHT = 0.5
TS_SURF_WEIGHT = 0.1
T_WEIGHT = 2
S_WEIGHT = 1
MIN_SEQ_DISTANCE = 10
MIN_3D_DISTANCE = 10
DIST_SEQ_WEIGHT = 1
DIST_3D_WEIGHT = 1

# Tien et al. 2013 max ASA (Å^2)
MAX_ASA = {
    'A':129, 'R':274, 'N':195, 'D':193, 'C':167, 'Q':225, 'E':223, 'G':104,
    'H':224, 'I':197, 'L':201, 'K':236, 'M':224, 'F':240, 'P':159, 'S':155,
    'T':172, 'W':285, 'Y':263, 'V':174
}

