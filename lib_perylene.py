# We offload all numerical work to numpy
import numpy as np

import sys
# for type hints import Self 
if sys.version_info.minor >= 11:
    from typing import Self
else:
    from typing_extensions import Self

import gvar as gv

import fitter

from fitModels import PriorBase, FitModelBase


# #############################################################################
# A few analytic results
# #############################################################################
# Each row gives (the amplitudes for the sites) for a particular irreducible single-particle operator.
amplitudes = np.array([
    [+0.1186437168889808, +0.1695811084523318, +0.1695811084523318, +0.1186437168889808, +0.1375120160641442, +0.3202939177295004, +0.3202939177295004, +0.1375120160641442, +0.2372874337779616, +0.3391622169046637, +0.3391622169046637, +0.2372874337779616, +0.1375120160641442, +0.3202939177295004, +0.3202939177295004, +0.1375120160641442, +0.1186437168889808, +0.1695811084523318, +0.1695811084523318, +0.1186437168889808, ],
    [-0.1835162557892201, +0.1481016933580872, +0.1481016933580872, -0.1835162557892201, +0.2523203780184878, -0.1396332470915334, -0.1396332470915334, +0.2523203780184878, -0.3670325115784402, +0.2962033867161745, +0.2962033867161745, -0.3670325115784402, +0.2523203780184878, -0.1396332470915334, -0.1396332470915334, +0.2523203780184878, -0.1835162557892201, +0.1481016933580872, +0.1481016933580872, -0.1835162557892201, ],
    [-0.1673410531263881, +0.0483574100924414, +0.0483574100924414, -0.1673410531263881, -0.3150286559866637, +0.2444024230451585, +0.2444024230451585, -0.3150286559866637, -0.3346821062527762, +0.0967148201848828, +0.0967148201848828, -0.3346821062527762, -0.3150286559866637, +0.2444024230451585, +0.2444024230451585, -0.3150286559866637, -0.1673410531263881, +0.0483574100924414, +0.0483574100924414, -0.1673410531263881, ],
    [+0.2236067977499789, +0.0000000000000000, +0.0000000000000000, +0.2236067977499789, -0.2236067977499789, -0.2236067977499789, -0.2236067977499789, -0.2236067977499789, +0.0000000000000000, +0.4472135954999579, +0.4472135954999579, +0.0000000000000000, -0.2236067977499789, -0.2236067977499789, -0.2236067977499789, -0.2236067977499789, +0.2236067977499789, +0.0000000000000000, +0.0000000000000000, +0.2236067977499789, ],
    [-0.2022599587389726, +0.3370999312316210, +0.3370999312316210, -0.2022599587389726, -0.1348399724926484, -0.1348399724926484, -0.1348399724926484, -0.1348399724926484, +0.3370999312316210, -0.0674199862463242, -0.0674199862463242, +0.3370999312316210, -0.1348399724926484, -0.1348399724926484, -0.1348399724926484, -0.1348399724926484, -0.2022599587389726, +0.3370999312316210, +0.3370999312316210, -0.2022599587389726, ],
    [-0.2886751345948128, -0.2886751345948128, -0.2886751345948128, -0.2886751345948128, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.2886751345948128, +0.2886751345948128, +0.2886751345948128, +0.2886751345948128, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, -0.2886751345948128, -0.2886751345948128, -0.2886751345948128, -0.2886751345948128, ],
    [-0.2142625365621797, -0.2886751345948128, -0.2886751345948128, -0.2142625365621797, -0.1140067144418895, -0.3282692510040693, -0.3282692510040693, -0.1140067144418895, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.1140067144418895, +0.3282692510040693, +0.3282692510040693, +0.1140067144418895, +0.2142625365621797, +0.2886751345948128, +0.2886751345948128, +0.2142625365621797, ],
    [-0.3282692510040693, +0.2886751345948128, +0.2886751345948128, -0.3282692510040693, +0.2142625365621797, -0.1140067144418895, -0.1140067144418895, +0.2142625365621797, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, -0.2142625365621797, +0.1140067144418895, +0.1140067144418895, -0.2142625365621797, +0.3282692510040693, -0.2886751345948128, -0.2886751345948128, +0.3282692510040693, ],
    [+0.2886751345948128, +0.0000000000000000, +0.0000000000000000, +0.2886751345948128, +0.2886751345948128, -0.2886751345948128, -0.2886751345948128, +0.2886751345948128, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, -0.2886751345948128, +0.2886751345948128, +0.2886751345948128, -0.2886751345948128, -0.2886751345948128, +0.0000000000000000, +0.0000000000000000, -0.2886751345948128, ],
    [+0.1140067144418895, +0.2886751345948128, +0.2886751345948128, +0.1140067144418895, -0.3282692510040693, -0.2142625365621797, -0.2142625365621797, -0.3282692510040693, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.3282692510040693, +0.2142625365621797, +0.2142625365621797, +0.3282692510040693, -0.1140067144418895, -0.2886751345948128, -0.2886751345948128, -0.1140067144418895, ],
    [+0.1186437168889808, -0.1695811084523318, +0.1695811084523318, -0.1186437168889808, +0.1375120160641442, -0.3202939177295004, +0.3202939177295004, -0.1375120160641442, +0.2372874337779616, -0.3391622169046637, +0.3391622169046637, -0.2372874337779616, +0.1375120160641442, -0.3202939177295004, +0.3202939177295004, -0.1375120160641442, +0.1186437168889808, -0.1695811084523318, +0.1695811084523318, -0.1186437168889808, ],
    [-0.1835162557892201, -0.1481016933580872, +0.1481016933580872, +0.1835162557892201, +0.2523203780184878, +0.1396332470915334, -0.1396332470915334, -0.2523203780184878, -0.3670325115784402, -0.2962033867161745, +0.2962033867161745, +0.3670325115784402, +0.2523203780184878, +0.1396332470915334, -0.1396332470915334, -0.2523203780184878, -0.1835162557892201, -0.1481016933580872, +0.1481016933580872, +0.1835162557892201, ],
    [-0.1673410531263881, -0.0483574100924414, +0.0483574100924414, +0.1673410531263881, -0.3150286559866637, -0.2444024230451585, +0.2444024230451585, +0.3150286559866637, -0.3346821062527762, -0.0967148201848828, +0.0967148201848828, +0.3346821062527762, -0.3150286559866637, -0.2444024230451585, +0.2444024230451585, +0.3150286559866637, -0.1673410531263881, -0.0483574100924414, +0.0483574100924414, +0.1673410531263881, ],
    [-0.2886751345948128, +0.2886751345948128, -0.2886751345948128, +0.2886751345948128, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.2886751345948128, -0.2886751345948128, +0.2886751345948128, -0.2886751345948128, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, -0.2886751345948128, +0.2886751345948128, -0.2886751345948128, +0.2886751345948128, ],
    [+0.2236067977499789, +0.0000000000000000, +0.0000000000000000, -0.2236067977499789, -0.2236067977499789, +0.2236067977499789, -0.2236067977499789, +0.2236067977499789, +0.0000000000000000, -0.4472135954999579, +0.4472135954999579, +0.0000000000000000, -0.2236067977499789, +0.2236067977499789, -0.2236067977499789, +0.2236067977499789, +0.2236067977499789, +0.0000000000000000, +0.0000000000000000, -0.2236067977499789, ],
    [+0.2022599587389726, +0.3370999312316210, -0.3370999312316210, -0.2022599587389726, +0.1348399724926484, -0.1348399724926484, +0.1348399724926484, -0.1348399724926484, -0.3370999312316210, -0.0674199862463242, +0.0674199862463242, +0.3370999312316210, +0.1348399724926484, -0.1348399724926484, +0.1348399724926484, -0.1348399724926484, +0.2022599587389726, +0.3370999312316210, -0.3370999312316210, -0.2022599587389726, ],
    [-0.2142625365621797, +0.2886751345948128, -0.2886751345948128, +0.2142625365621797, -0.1140067144418895, +0.3282692510040693, -0.3282692510040693, +0.1140067144418895, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.1140067144418895, -0.3282692510040693, +0.3282692510040693, -0.1140067144418895, +0.2142625365621797, -0.2886751345948128, +0.2886751345948128, -0.2142625365621797, ],
    [-0.3282692510040693, -0.2886751345948128, +0.2886751345948128, +0.3282692510040693, +0.2142625365621797, +0.1140067144418895, -0.1140067144418895, -0.2142625365621797, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, -0.2142625365621797, -0.1140067144418895, +0.1140067144418895, +0.2142625365621797, +0.3282692510040693, +0.2886751345948128, -0.2886751345948128, -0.3282692510040693, ],
    [+0.2886751345948128, +0.0000000000000000, +0.0000000000000000, -0.2886751345948128, +0.2886751345948128, +0.2886751345948128, -0.2886751345948128, -0.2886751345948128, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, -0.2886751345948128, -0.2886751345948128, +0.2886751345948128, +0.2886751345948128, -0.2886751345948128, +0.0000000000000000, +0.0000000000000000, +0.2886751345948128, ],
    [+0.1140067144418895, -0.2886751345948128, +0.2886751345948128, -0.1140067144418895, -0.3282692510040693, +0.2142625365621797, -0.2142625365621797, +0.3282692510040693, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.0000000000000000, +0.3282692510040693, -0.2142625365621797, +0.2142625365621797, -0.3282692510040693, -0.1140067144418895, +0.2886751345948128, -0.2886751345948128, +0.1140067144418895, ],
])

# Each row gives the quantum numbers of a single irreducible single-particle operator.
# The operators are in the same order as above.
#
# The 0th and 1st columns give eigenvalues under the following symmetries
#   <-- 0 -->
#
#   O       O   ^
#       O       1
#   O       O   v
#
# and the 2nd column is the eigenvalue of the hopping matrix.
quantum_numbers = np.array([
    [+1.0000000000000000, +1.0000000000000000, -2.5883639906851044, 0],
    [+1.0000000000000000, +1.0000000000000000, +2.1819433360523916, 1],
    [+1.0000000000000000, +1.0000000000000000, -1.5935793453672884, 2],
    [+1.0000000000000000, +1.0000000000000000, +1.0000000000000002, 3],
    [+1.0000000000000000, +1.0000000000000000, +1.0000000000000000, 4],
    [+1.0000000000000000, +1.0000000000000000, -1.0000000000000002, 5],
                                                                   
    [-1.0000000000000000, +1.0000000000000000, -1.8793852415718164, 0],
    [-1.0000000000000000, +1.0000000000000000, +1.5320888862379560, 1],
    [-1.0000000000000000, +1.0000000000000000, -1.0000000000000002, 2],
    [-1.0000000000000000, +1.0000000000000000, +0.3472963553338607, 3],
                                                                   
    [+1.0000000000000000, -1.0000000000000000, +2.5883639906851044, 0],
    [+1.0000000000000000, -1.0000000000000000, -2.1819433360523916, 1],
    [+1.0000000000000000, -1.0000000000000000, +1.5935793453672884, 2],
    [+1.0000000000000000, -1.0000000000000000, +1.0000000000000002, 3],
    [+1.0000000000000000, -1.0000000000000000, -1.0000000000000002, 4],
    [+1.0000000000000000, -1.0000000000000000, -1.0000000000000000, 5],
                                                                   
    [-1.0000000000000000, -1.0000000000000000, +1.8793852415718164, 0],
    [-1.0000000000000000, -1.0000000000000000, -1.5320888862379562, 1],
    [-1.0000000000000000, -1.0000000000000000, +1.0000000000000002, 2],
    [-1.0000000000000000, -1.0000000000000000, -0.3472963553338607, 3],
])
# we invert the list to match the hh^+ operators

D2irreps = np.array([
    r"$A^0$",   # 0
    r"$A^1$",   # 1
    r"$A^2$",   # 2
    r"$A^3$",   # 3
    r"$A^4$",   # 4
    r"$A^5$",   # 5
                                                                   
    r"$B_1^0$", # 6
    r"$B_1^1$", # 7
    r"$B_1^2$", # 8
    r"$B_1^3$", # 9
                                                                   
    r"$B_2^0$", # 10
    r"$B_2^1$", # 11
    r"$B_2^2$", # 12
    r"$B_2^3$", # 13
    r"$B_2^4$", # 14
    r"$B_2^5$", # 15
                
    r"$B_3^0$", # 16
    r"$B_3^1$", # 17
    r"$B_3^2$", # 18
    r"$B_3^3$", # 19
]) 
# a slice object for each irrep from above
slice_pp = np.s_[0:6] 
slice_mp = np.s_[6:10] 
slice_pm = np.s_[10:16] 
slice_mm = np.s_[16:20] 


decayingCorr = {
    "mu0/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.1/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.2/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.3/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.4/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.5/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.6/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.7/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.8/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.9/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu1/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu1.1/beta4": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": True,  
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    # ===============================================================
    "mu0/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.1/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.2/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.3/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.4/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.5/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.6/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.7/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.8/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.9/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu1/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu1.1/beta6": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": True,  
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    # ===============================================================
    "mu0/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.1/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.2/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.3/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.4/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": False, 
    },
    "mu0.5/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.6/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.7/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.8/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu0.9/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu1/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": False, 
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
    "mu1.1/beta8": {
        "irrepID0" : False,
        "irrepID1" : True,  
        "irrepID2" : False,  
        "irrepID3" : True, 
        "irrepID4" : True,
        "irrepID5" : False,
        "irrepID6" : False,
        "irrepID7" : True,
        "irrepID8" : False, 
        "irrepID9" : True, 
        "irrepID10": True, 
        "irrepID11": False, 
        "irrepID12": True, 
        "irrepID13": True, 
        "irrepID14": True,  
        "irrepID15": False,
        "irrepID16": True, 
        "irrepID17": False, 
        "irrepID18": True, 
        "irrepID19": True, 
    },
}


# #############################################################################
# Model & Prior for the fitting
# #############################################################################

class DoubleExponentialsModel(FitModelBase): 
    def __init__(self, 
        Nstates_D:int, 
        Nstates_I:int, 
        Nt: int, 
        beta: float
    ):
        self.Nstates_D = Nstates_D
        self.Nstates_I = Nstates_I
        self.beta = beta
        self.Nt = Nt
        self.delta = beta/Nt

    def __call__(self,tau:np.ndarray, p:dict) -> np.ndarray:
        r"""
            Arguments:
                tau: np.array(Nt), abscissa in lattice units, e.g. `np.arange(Nt)` or `np.linspace(0,T,Nt)/delta`
                p: dict, dictionary of parameters. Keys are assumed to be 
                    - A0, A1, ... (overlap factors)
                    - E0, E1, ... (energies in lattice units)
            
            Evaluating the function 

            \f[
                C(t) &= \sum_n A_n e^{- t E_n} + B_n e^{ t F_n} \\
                     &= A_0 e^{- t E_0} + B_0 e^{ t F_0} 
                      + \sum_{n>0} A_n e^{- t ΔE_n} + B_n e^{ t ΔF_n}
            \f]
        """
        out = np.zeros_like(tau, dtype = object)
        
        E = p["E0"]
        F = p["F0"]

        out += p["A0"] * np.exp( - tau * E * self.delta )
        out += p["B0"] * np.exp((tau-self.Nt) * F * self.delta ) 

        for n in range(1,self.Nstates_D):
            E += p[f"ΔE{n}"]
            out += p[f"A{n}"] * np.exp( - tau * E * self.delta )

        for n in range(1,self.Nstates_I):
            F += p[f"ΔF{n}"]
            out += p[f"B{n}"] * np.exp((tau-self.Nt) * F * self.delta )

        return out

    def __repr__(self):
        return f"DoubleExpModel[Nstates=({self.Nstates_D},{self.Nstates_I})]"

class DoubleExponentialsDataBasedPrior(PriorBase): 
    def __init__(self, 
        Nstates_D:int, 
        Nstates_I:int, 
        irrepID:int,
        CLIargs:object,
        signMu:float,
        modelAverage:list[object] = None,
        repr = "Prior"
    ):
        self.Nstates_D = Nstates_D
        self.Nstates_I = Nstates_I
        self.irrepID = irrepID
        self.signMu = signMu
        self.args = CLIargs
        self.repr = repr
        self.perc = 0.1
        self.width = 5
        self.modelAverage = modelAverage

    def flatPrior(self):
        prior = gv.BufferDict()

        if self.modelAverage is None:
            EnonInt = np.abs( quantum_numbers[self.irrepID,2] + self.signMu * self.args.mu )
            # In case it is exactly 0 put something small
            if EnonInt < 1e-14:
                prior[f"E{0}"] = gv.gvar(0,10)
                prior[f"F{0}"] = gv.gvar(0, 10)
            else:
                prior[f"log(E{0})"] = gv.log( gv.gvar(EnonInt, EnonInt) )
                prior[f"log(F{0})"] = gv.log( gv.gvar(EnonInt, EnonInt) )

            prior[f"A{0}"] = gv.gvar(1,1)
            prior[f"B{0}"] = gv.gvar(1,1)

            for n in range(1,self.Nstates_D):
                prior[f"log(ΔE{n})"] = gv.log(gv.gvar(0.5, 1))
                prior[f"A{n}"] = gv.gvar(1, 10)

            for n in range(1,self.Nstates_I):
                prior[f"log(ΔF{n})"] = gv.log(gv.gvar(0.5, 1))
                prior[f"B{n}"] = gv.gvar(1, 10)

        else:
            prior[f"log(E{0})"] = gv.log(gv.gvar(
                self.modelAverage['est']['E0'], 
                np.maximum(
                    self.perc *self.modelAverage['est']['E0'],
                    self.width*self.modelAverage['err']['E0']
                )
            ))
            prior[f"log(F{0})"] = gv.log(gv.gvar(
                self.modelAverage['est']['F0'], 
                np.maximum(
                    self.perc *self.modelAverage['est']['F0'],
                    self.width*self.modelAverage['err']['F0']
                )
            ))
            prior[f"A{0}"] =gv.gvar(
                self.modelAverage['est']['A0'], 
                np.maximum(
                    self.perc *self.modelAverage['est']['A0'],
                    self.width*self.modelAverage['err']['A0']
                )
            )
            prior[f"B{0}"] =gv.gvar(
                self.modelAverage['est']['B0'], 
                np.maximum(
                    self.perc *self.modelAverage['est']['B0'],
                    self.width*self.modelAverage['err']['B0']
                )
            )

            for n in range(1,self.Nstates_D-1):
                prior[f"log(ΔE{n})"] = gv.log(gv.gvar(
                    self.modelAverage['est'][f'ΔE{n}'], 
                    np.maximum(
                        self.perc *self.modelAverage['est'][f'ΔE{n}'],
                        self.width*self.modelAverage['err'][f'ΔE{n}']
                    )
                ))
                prior[f"A{n}"] = gv.gvar(
                    self.modelAverage['est'][f'A{n}'], 
                    np.maximum(
                        self.perc *self.modelAverage['est'][f'A{n}'],
                        self.width*self.modelAverage['err'][f'A{n}']
                    )
                )
            for n in range(1,self.Nstates_I-1):
                prior[f"log(ΔF{n})"] = gv.log(gv.gvar(
                    self.modelAverage['est'][f'ΔF{n}'], 
                    np.maximum(
                        self.perc *self.modelAverage['est'][f'ΔF{n}'],
                        self.width*self.modelAverage['err'][f'ΔF{n}']
                    )
                ))
                prior[f"B{n}"] = gv.gvar(
                    self.modelAverage['est'][f'B{n}'], 
                    np.maximum(
                        self.perc *self.modelAverage['est'][f'B{n}'],
                        self.width*self.modelAverage['err'][f'B{n}']
                    )
                )

            # and flat for the new values
            if self.Nstates_D-1 > 0:
                prior[f"log(ΔE{self.Nstates_D-1})"] = gv.log(gv.gvar(0.5, 1))
                prior[f"A{self.Nstates_D-1}"] = gv.gvar(1, 1)

            if self.Nstates_I-1 > 0:
                prior[f"log(ΔF{self.Nstates_I-1})"] = gv.log(gv.gvar(0.5, 1))
                prior[f"B{self.Nstates_I-1}"] = gv.gvar(1, 1)

        return prior

    def __call__(self, nbst = None) -> gv.BufferDict:
        prior = self.flatPrior()
        
        # resample the mean on each bootstrap sample
        if nbst is not None:
            for key in prior.keys():
                prior[key] = gv.gvar(
                    gv.sample(prior[key],1), 
                    prior[key].sdev
                )

        return prior

    def __repr__(self):
        return self.repr
