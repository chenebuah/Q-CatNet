## CONTRIBUTORS: * Ericsson Chenebuah, Michel Nganbe, David Liu and Alain Tchagang
# Department of Mechanical Engineering, University of Ottawa, 75 Laurier Ave. East, Ottawa, ON, K1N 6N5 Canada
# Digital Technologies Research Centre, National Research Council of Canada, 1200 Montréal Road, Ottawa, ON, K1A 0R6 Canada
# * email: echen013@uottawa.ca

# Saved Pytorch Models can be downloaded from the "Saved Models Folder".
# Learning Models Codes can be found in the "Learning Models" Folder.

#Intall libraries and packages:

#! pip install pymatgen
#! pip install mp_api
#! pip install ase==3.20.1
#! pip install dscribe
#! pip install torch_geometric
#! pip install scikit-optimize

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
import random
from ase.neighborlist import NeighborList
from ase import Atoms, neighborlist
import tensorflow as tf
from ase.geometry import get_angles
from dscribe.descriptors import CoulombMatrix, EwaldSumMatrix, SineMatrix
from pymatgen.core import Molecule
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Dataset, Data
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from sklearn.preprocessing import MinMaxScaler
from mp_api.client import MPRester

atomic_no = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10, "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18, "K":19, "Ca":20,
    "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27, "Ni":28, "Cu":29, "Zn":30, "Ga":31, "Ge":32, "As":33, "Se":34, "Br":35, "Kr":36, "Rb":37, "Sr":38,
    "Y":39, "Zr":40, "Nb":41, "Mo":42, "Tc":43, "Ru":44, "Rh":45, "Pd":46, "Ag":47, "Cd":48, "In":49, "Sn":50, "Sb":51, "Te":52, "I":53, "Xe":54, "Cs":55, "Ba":56,
    "La":57, "Ce":58, "Pr":59, "Nd":60, "Pm":61, "Sm":62, "Eu":63, "Gd":64, "Tb":65, "Dy":66, "Ho":67, "Er":68, "Tm":69, "Yb":70, "Lu":71, "Hf":72, "Ta":73, "W":74,
    "Re":75, "Os":76, "Ir":77, "Pt":78, "Au":79, "Hg":80, "Tl":81, "Pb":82, "Bi":83, "Po":84, "At":85, "Rn":86, "Fr":87, "Ra":88, "Ac":89, "Th":90, "Pa":91, "U":92,
    "Np":93, "Pu":94, "Am":95, "Cm":96, "Bk":97, "Cf":98, "Es":99, "Fm":100, "Md":101, "No":102, "Lr":103
    }
group_no = {
    "H":1, "He":18, "Li":1, "Be":2, "B":13, "C":14, "N":15, "O":16, "F":17, "Ne":18, "Na":1, "Mg":2, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18, "K":1, "Ca":2,
    "Sc":3, "Ti":4, "V":5, "Cr":6, "Mn":7, "Fe":8, "Co":9, "Ni":10, "Cu":11, "Zn":12, "Ga":13, "Ge":14, "As":15, "Se":16, "Br":17, "Kr":18, "Rb":1, "Sr":2,
    "Y":3, "Zr":4, "Nb":5, "Mo":6, "Tc":7, "Ru":8, "Rh":9, "Pd":10, "Ag":11, "Cd":12, "In":13, "Sn":14, "Sb":15, "Te":16, "I":17, "Xe":18, "Cs":1, "Ba":2,
    "La":3, "Ce":4, "Pr":5, "Nd":6, "Pm":7, "Sm":8, "Eu":9, "Gd":10, "Tb":11, "Dy":12, "Ho":13, "Er":14, "Tm":15, "Yb":16, "Lu":17, "Hf":4, "Ta":5, "W":6,
    "Re":7, "Os":8, "Ir":9, "Pt":10, "Au":11, "Hg":12, "Tl":13, "Pb":14, "Bi":15, "Po":16, "At":17, "Rn":18, "Fr":1, "Ra":2, "Ac":3, "Th":4, "Pa":5, "U":6,
    "Np":7, "Pu":8, "Am":9, "Cm":10, "Bk":11, "Cf":12, "Es":13, "Fm":14, "Md":15, "No":16, "Lr":17
    }
row_no = {
    "H":1, "He":1, "Li":2, "Be":2, "B":2, "C":2, "N":2, "O":2, "F":2, "Ne":2, "Na":3, "Mg":3, "Al":3, "Si":3, "P":3, "S":3, "Cl":3, "Ar":3, "K":4, "Ca":4,
    "Sc":4, "Ti":4, "V":4, "Cr":4, "Mn":4, "Fe":4, "Co":4, "Ni":4, "Cu":4, "Zn":4, "Ga":4, "Ge":4, "As":4, "Se":4, "Br":4, "Kr":4, "Rb":5, "Sr":5,
    "Y":5, "Zr":5, "Nb":5, "Mo":5, "Tc":5, "Ru":5, "Rh":5, "Pd":5, "Ag":5, "Cd":5, "In":5, "Sn":5, "Sb":5, "Te":5, "I":5, "Xe":5, "Cs":6, "Ba":6,
    "La":8, "Ce":8, "Pr":8, "Nd":8, "Pm":8, "Sm":8, "Eu":8, "Gd":8, "Tb":8, "Dy":8, "Ho":8, "Er":8, "Tm":8, "Yb":8, "Lu":8, "Hf":6, "Ta":6, "W":6,
    "Re":6, "Os":6, "Ir":6, "Pt":6, "Au":6, "Hg":6, "Tl":6, "Pb":6, "Bi":6, "Po":6, "At":6, "Rn":6, "Fr":7, "Ra":7, "Ac":9, "Th":9, "Pa":9, "U":9,
    "Np":9, "Pu":9, "Am":9, "Cm":9, "Bk":9, "Cf":9, "Es":9, "Fm":9, "Md":9, "No":9, "Lr":9
    }
valence = {
    "H":1, "He":2, "Li":1, "Be":2, "B":3, "C":4, "N":5, "O":6, "F":7, "Ne":8, "Na":1, "Mg":2, "Al":3, "Si":4, "P":5, "S":6, "Cl":7, "Ar":8, "K":1, "Ca":2,
    "Sc":2, "Ti":2, "V":2, "Cr":1, "Mn":2, "Fe":2, "Co":2, "Ni":2, "Cu":1, "Zn":2, "Ga":3, "Ge":4, "As":5, "Se":6, "Br":7, "Kr":8, "Rb":1, "Sr":2,
    "Y":2, "Zr":2, "Nb":1, "Mo":1, "Tc":2, "Ru":1, "Rh":1, "Pd":9, "Ag":1, "Cd":2, "In":3, "Sn":4, "Sb":5, "Te":6, "I":7, "Xe":8, "Cs":1, "Ba":2,
    "La":2, "Ce":2, "Pr":2, "Nd":2, "Pm":2, "Sm":2, "Eu":2, "Gd":2, "Tb":2, "Dy":2, "Ho":2, "Er":2, "Tm":2, "Yb":2, "Lu":2, "Hf":2, "Ta":2, "W":2,
    "Re":2, "Os":2, "Ir":2, "Pt":1, "Au":1, "Hg":2, "Tl":3, "Pb":4, "Bi":5, "Po":6, "At":7, "Rn":8, "Fr":1, "Ra":2, "Ac":2, "Th":2, "Pa":2, "U":2,
    "Np":2, "Pu":2, "Am":2, "Cm":2, "Bk":2, "Cf":2, "Es":2, "Fm":2, "Md":2, "No":2, "Lr":3
    }
block = {
    "H":1, "He":1, "Li":1, "Be":1, "B":2, "C":2, "N":2, "O":2, "F":2, "Ne":2, "Na":1, "Mg":1, "Al":2, "Si":2, "P":2, "S":2, "Cl":2, "Ar":2, "K":1, "Ca":1,
    "Sc":3, "Ti":3, "V":3, "Cr":3, "Mn":3, "Fe":3, "Co":3, "Ni":3, "Cu":3, "Zn":3, "Ga":2, "Ge":2, "As":2, "Se":2, "Br":2, "Kr":2, "Rb":1, "Sr":1,
    "Y":3, "Zr":3, "Nb":3, "Mo":3, "Tc":3, "Ru":3, "Rh":3, "Pd":3, "Ag":3, "Cd":3, "In":2, "Sn":2, "Sb":2, "Te":2, "I":2, "Xe":2, "Cs":1, "Ba":1,
    "La":4, "Ce":4, "Pr":4, "Nd":4, "Pm":4, "Sm":4, "Eu":4, "Gd":4, "Tb":4, "Dy":4, "Ho":4, "Er":4, "Tm":4, "Yb":4, "Lu":3, "Hf":3, "Ta":3, "W":3,
    "Re":3, "Os":3, "Ir":3, "Pt":3, "Au":3, "Hg":3, "Tl":2, "Pb":2, "Bi":2, "Po":2, "At":2, "Rn":2, "Fr":1, "Ra":1, "Ac":4, "Th":4, "Pa":4, "U":4,
    "Np":4, "Pu":4, "Am":4, "Cm":4, "Bk":4, "Cf":4, "Es":4, "Fm":4, "Md":4, "No":4, "Lr":3
    }
electronegativity = {
    'H':2.2, 'He':0.7, 'Li':0.98, 'Be':	1.57, 'B':	2.04, 'C':2.55,'N':	3.04,'O':	3.44,'F':	3.98,'Ne':	0,'Na':	0.93,'Mg':	1.31,'Al':	1.61,'Si':	1.9,
    'P':	2.19,'S':	2.58,'Cl':	3.16,'Ar':	0, 'K':	0.82, 'Ca':	1, 'Sc':	1.36, 'Ti':	1.54, 'V':	1.63,'Cr':	1.66,'Mn':	1.55,'Fe':	1.83,'Co':	1.88,'Ni':	1.91,
    'Cu':	1.9, 'Zn':	1.65, 'Ga':	1.81, 'Ge':	2.01,'As':	2.18,'Se':	2.55,'Br':	2.96,'Kr':	0,'Rb':	0.82,'Sr':	0.95,'Y':	1.22,'Zr':	1.33,'Nb':	1.6,'Mo':	2.16,
    'Tc':	2.1,'Ru':	2.2,'Rh':	2.28,'Pd':	2.2,'Ag':	1.93,'Cd':	1.69,'In':	1.78,'Sn':	1.96,'Sb':	2.05,'Te':	2.1,'I':	2.66,'Xe':	2.6,'Cs':	0.79,'Ba':	0.89,
    'La':	1.1,'Ce':	1.12,'Pr':	1.13,'Nd':	1.14,'Pm':	1.13,'Sm':	1.17,'Eu':	1.2,'Gd':	1.2,'Tb':	1.1,'Dy':	1.22,'Ho':	1.23,'Er':	1.24,'Tm':	1.25,'Yb':	1.1,
    'Lu':	1,'Hf':	1.3,'Ta':	1.5,'W':	1.7,'Re':	1.9,'Os':	2.2,'Ir':	2.2,'Pt':	2.2,'Au':	2.4,'Hg':	1.9,'Tl':	1.8,'Pb':	1.8,'Bi':	1.9,'Po':	2,'At':	2.2,'Rn':	2.2,'Fr':	0.7,
    'Ra':	0.9,'Ac':	1.1,'Th':	1.3,'Pa':	1.5,'U':	1.7,'Np':	1.3,'Pu':	1.3,'Am':	1.3,'Cm':	1.3,'Bk':	1.3,'Cf':	1.3,'Es':	1.3,'Fm':	1.3,'Md':	1.3,'No':	1.3,'Lr':	1.3
    }
covalent_radius = {
    'H':	0.31,'He':	0.28,'Li':	1.28,'Be':	0.96,'B':	0.84,'C':	0.76,'N':	0.71,'O':	0.66,'F':	0.57,'Ne':	0.58,'Na':	1.66,'Mg':	1.41,'Al':	1.21,'Si':	1.11,'P':	1.07,
    'S':	1.05,'Cl':	1.02,'Ar':	1.06,'K':	2.03,'Ca':	1.76,'Sc':	1.7,'Ti':	1.6,'V':	1.53,'Cr':	1.39,'Mn':	1.39,'Fe':	1.32,'Co':	1.26,'Ni':	1.24,'Cu':	1.32,'Zn':	1.22,
    'Ga':	1.22,'Ge':	1.2,'As':	1.19,'Se':	1.2,'Br':	1.2,'Kr':	1.16,'Rb':	2.2,'Sr':	1.95,'Y':	1.9,'Zr':	1.75,'Nb':	1.64,'Mo':	1.54,'Tc':	1.47,'Ru':	1.46,'Rh':	1.42,
    'Pd':	1.39,'Ag':	1.45,'Cd':	1.44,'In':	1.42,'Sn':	1.39,'Sb':	1.39,'Te':	1.38,'I':	1.39,'Xe':	1.4,'Cs':	2.44,'Ba':	2.15,'La':	2.07,'Ce':	2.04,'Pr':	2.03,'Nd':	2.01,
    'Pm':	1.99,'Sm':	1.98,'Eu':	1.98,'Gd':	1.96,'Tb':	1.94,'Dy':	1.92,'Ho':	1.92,'Er':	1.89,'Tm':	1.9,'Yb':	1.87,'Lu':	1.87,'Hf':	1.75,'Ta':	1.7,'W':	1.62,'Re':	1.51,
    'Os':	1.44,'Ir':	1.41,'Pt':	1.36,'Au':	1.36,'Hg':	1.32,'Tl':	1.45,'Pb':	1.46,'Bi':	1.48,'Po':	1.4,'At':	1.5,'Rn':	1.5,'Fr':	2.6,'Ra':	2.21,'Ac':	2.15,'Th':	2.06,
    'Pa':	2,'U':	1.96,'Np':	1.9,'Pu':	1.87,'Am':	1.8,'Cm':	1.69,'Bk':	'n.a','Cf':	'n.a','Es':	'n.a','Fm':	'n.a','Md':	'n.a','No':	'n.a','Lr':	'n.a'
    }
ionization_energy = {
    'H':	13.59844,'He':	24.58741,'Li':	5.39172,'Be':	9.3227,'B':	8.29803,'C':	11.2603,'N':	14.53414,'O':	13.61806,'F':	17.42282,'Ne':	21.5646,'Na':	5.13908,'Mg':	7.64624,
    'Al':	5.98577,'Si':	8.15169,'P':	10.48669,'S':	10.36001,'Cl':	12.96764,'Ar':	15.75962,'K':	4.34066,'Ca':	6.11316,'Sc':	6.5615,'Ti':	6.8281,'V':	6.7462,'Cr':	6.7665,
    'Mn':	7.43402,'Fe':	7.9024,'Co':	7.881,'Ni':	7.6398,'Cu':	7.72638,'Zn':	9.3942,'Ga':	5.9993,'Ge':	7.8994,'As':	9.7886,'Se':	9.75238,'Br':	11.81381,'Kr':	13.99961,
    'Rb':	4.17713,'Sr':	5.6949,'Y':	6.2171,'Zr':	6.6339,'Nb':	6.75885,'Mo':	7.09243,'Tc':	7.28,'Ru':	7.3605,'Rh':	7.4589,'Pd':	8.3369,'Ag':	7.5762,'Cd':	8.9938,'In':	5.78636,
    'Sn':	7.3439,'Sb':	8.6084,'Te':	9.0096,'I':	10.45126,'Xe':	12.1298,'Cs':	3.8939,'Ba':	5.2117,'La':	5.5769,'Ce':	5.5387,'Pr':	5.473,'Nd':	5.525,'Pm':	5.582,'Sm':	5.6436,
    'Eu':	5.6704,'Gd':	6.1501,'Tb':	5.8638,'Dy':	5.9389,'Ho':	6.0215,'Er':	6.1077,'Tm':	6.18431,'Yb':	6.25416,'Lu':	5.4259,'Hf':	6.82507,'Ta':	7.5496,'W':	7.864,'Re':	7.8335,
    'Os':	8.4382,'Ir':	8.967,'Pt':	8.9587,'Au':	9.2255,'Hg':	10.4375,'Tl':	6.1082,'Pb':	7.41666,'Bi':	7.2856,'Po':	8.417,'At':	2.2,'Rn':	10.7485,'Fr':	4.0727,'Ra':	5.2784,
    'Ac':	5.17,'Th':	6.3067,'Pa':	5.89,'U':	6.19405,'Np':	6.2657,'Pu':	6.0262,'Am':	5.9738,'Cm':	5.9915,'Bk':	6.1979,'Cf':	6.2817,'Es':	6.42,'Fm':	6.5,'Md':	6.58,'No':	6.65,
    'Lr':	4.9
    }
electron_affinity = {
    'H':	0.754195,'He':	-0.5182,'Li':	0.618049,'Be':	-0.5182,'B':	0.279723,'C':	1.262119,'N':	-0.0725,'O':	1.4611096,'F':	3.4011895,'Ne':	-1.2437,'Na':	0.547926,'Mg':	-0.4146,
    'Al':	0.43283,'Si':	1.389522,'P':	0.7465,'S':	2.077103,'Cl':	3.612724,'Ar':	-0.995,'K':	0.50147,'Ca':	0.02455,'Sc':	0.188,'Ti':	0.079,'V':	0.525,'Cr':	0.666,'Mn':	-0.5,
    'Fe':	0.151,'Co':	0.662,'Ni':	1.156,'Cu':	1.235,'Zn':	-0.6219,'Ga':	0.43,'Ge':	1.232712,'As':	0.814,'Se':	2.02067,'Br':	3.363588,'Kr':	-1,'Rb':	0.48592,'Sr':	0.048,'Y':	0.307,
    'Zr':	0.426,'Nb':	0.8933,'Mo':	0.748,'Tc':	0.55,'Ru':	1.05,'Rh':	1.137,'Pd':	0.562,'Ag':	1.302,'Cd':	-0.7255,'In':	0.3,'Sn':	1.112067,'Sb':	1.046,'Te':	1.9708,'I':	3.059037,
    'Xe':	-0.8291,'Cs':	0.471626,'Ba':	0.14462,'La':	0.47,'Ce':	0.57,'Pr':	0.964,'Nd':	0.09749,'Pm':	0.124,'Sm':	0.166,'Eu':	0.114,'Gd':	0.135,'Tb':	1.161,'Dy':	0.352,'Ho':	0.342,
    'Er':	0.311,'Tm':	1.03,'Yb':	-0.02,'Lu':	0.34,'Hf':	0.176,'Ta':	0.322,'W':	0.815,'Re':	0.15,'Os':	1.1,'Ir':	1.5638,'Pt':	2.128,'Au':	2.30863,'Hg':	-0.5182,'Tl':	0.2,'Pb':	0.364,
    'Bi':	0.946,'Po':	1.9,'At':	2.8,'Rn':	-0.7255,'Fr':	0.46,'Ra':	0.1,'Ac':	0.35,'Th':	0.60769,'Pa':	0.57,'U':	0.31497,'Np':	0.477,'Pu':	-0.5,'Am':	0.1,'Cm':	0.28,'Bk':	-1.72,
    'Cf':	-1.01,'Es':	-0.3,'Fm':	0.35,'Md':	0.98,'No':	-2.33,'Lr':	-0.31
    }
molar_volume = {
    'H':	11.42,'He':	21,'Li':	13.02,'Be':	4.85,'B':	4.39,'C':	5.29,'N':	13.54,'O':	17.36,'F':	11.2,'Ne':	13.23,'Na':	23.78,'Mg':	14,'Al':	10,'Si':	12.06,'P':	17.02,'S':	15.53,
    'Cl':	17.39,'Ar':	22.56,'K':	45.94,'Ca':	26.2,'Sc':	15,'Ti':	10.64,'V':	8.32,'Cr':	7.23,'Mn':	7.35,'Fe':	7.09,'Co':	6.67,'Ni':	6.59,'Cu':	7.11,'Zn':	9.16,'Ga':	11.8,
    'Ge':	13.63,'As':	12.95,'Se':	16.42,'Br':	19.78,'Kr':	27.99,'Rb':	55.76,'Sr':	33.94,'Y':	19.88,'Zr':	14.02,'Nb':	10.83,'Mo':	9.38,'Tc':	8.63,'Ru':	8.17,'Rh':	8.28,'Pd':	8.56,
    'Ag':	10.27,'Cd':	13,'In':	15.76,'Sn':	16.29,'Sb':	18.19,'Te':	20.46,'I':	25.72,'Xe':	35.92,'Cs':	70.94,'Ba':	38.16,'La':	22.39,'Ce':	20.69,'Pr':	20.8,'Nd':	20.59,'Pm':	20.23,
    'Sm':	19.98,'Eu':	28.97,'Gd':	19.9,'Tb':	19.3,'Dy':	19.01,'Ho':	18.74,'Er':	18.46,'Tm':	19.1,'Yb':	24.84,'Lu':	17.78,'Hf':	13.44,'Ta':	10.85,'W':	9.47,'Re':	8.86,'Os':	8.42,
    'Ir':	8.52,'Pt':	9.09,'Au':	10.21,'Hg':	14.09,'Tl':	17.22,'Pb':	18.26,'Bi':	21.31,'Po':	22.97,'At':	23.6,'Rn':	50.5,'Fr':	77,'Ra':	41.09,'Ac':	22.55,'Th':	19.8,'Pa':	15.18,
    'U':	12.49,'Np':	11.59,'Pu':	12.29,'Am':	17.63,'Cm':	18.05,'Bk':	16.84,'Cf':	16.5,'Es':	28.52,'Fm':	"n.a",'Md':	"n.a",'No':	"n.a",'Lr':	"n.a"
    }
average_ionic_radius = {
    'H':	0,'He':	0,'Li':	0.9,'Be':	0.59,'B':	0.41,'C':	0.3,'N':	0.63,'O':	1.26,'F':	0.705,'Ne':	0,'Na':	1.16,'Mg':	0.86,'Al':	0.675,'Si':	0.54,'P':	0.55,'S':	0.88,'Cl':	0.78,
    'Ar':	0,'K':	1.52,'Ca':	1.14,'Sc':	0.885,'Ti':	0.852,'V':	0.777,'Cr':	0.94,'Mn':	0.648,'Fe':	0.853,'Co':	0.768,'Ni':	0.74,'Cu':	0.82,'Zn':	0.88,'Ga':	0.76,'Ge':	0.77,
    'As':	0.66,'Se':	1.013,'Br':	0.883,'Kr':	0,'Rb':	1.66,'Sr':	1.32,'Y':	1.04,'Zr':	0.86,'Nb':	0.82,'Mo':	0.775,'Tc':	0.742,'Ru':	0.661,'Rh':	0.745,'Pd':	0.846,'Ag':	1.087,
    'Cd':	1.09,'In':	0.94,'Sn':	0.83,'Sb':	0.83,'Te':	1.293,'I':	1.273,'Xe':	0.62,'Cs':	1.81,'Ba':	1.49,'La':	1.172,'Ce':	1.08,'Pr':	1.06,'Nd':	1.276,'Pm':	1.11,'Sm':	1.229,
    'Eu':	1.199,'Gd':	1.075,'Tb':	0.982,'Dy':	1.131,'Ho':	1.041,'Er':	1.03,'Tm':	1.095,'Yb':	1.084,'Lu':	1.001,'Hf':	0.85,'Ta':	0.82,'W':	0.767,'Re':	0.712,'Os':	0.673,'Ir':	0.765,
    'Pt':	0.805,'Au':	1.07,'Hg':	1.245,'Tl':	1.333,'Pb':	1.123,'Bi':	1.035,'Po':	0.945,'At':	0.76,'Rn':	0,'Fr':	1.94,'Ra':	1.62,'Ac':	1.26,'Th':	1.08,'Pa':	1.04,'U':	0.991,'Np':	1,
    'Pu':	0.967,'Am':	1.168,'Cm':	1.05,'Bk':	1.035,'Cf':	1.026,'Es':	0,'Fm':	0,'Md':	0,'No':	0,'Lr':	0
    }
polarizability = {
    'H':	0.666793,'He':	0.204956,'Li':	24.3,'Be':	5.6,'B':	3.03,'C':	1.76,'N':	1.1,'O':	0.802,'F':	0.557,'Ne':	0.3956,'Na':	24.11,'Mg':	10.6,'Al':	6.8,'Si':	5.38,'P':	3.63,
    'S':	2.9,'Cl':	2.18,'Ar':	1.6411,'K':	43.4,'Ca':	22.8,'Sc':	17.8,'Ti':	14.6,'V':	12.4,'Cr':	11.6,'Mn':	9.4,'Fe':	8.4,'Co':	7.5,'Ni':	6.8,'Cu':	6.2,'Zn':	5.75,'Ga':	8.12,
    'Ge':	6.07,'As':	4.31,'Se':	3.77,'Br':	3.05,'Kr':	2.4844,'Rb':	47.3,'Sr':	27.6,'Y':	22.7,'Zr':	17.9,'Nb':	15.7,'Mo':	12.8,'Tc':	11.4,'Ru':	9.6,'Rh':	8.6,'Pd':	4.8,
    'Ag':	7.2,'Cd':	7.36,'In':	10.2,'Sn':	7.7,'Sb':	6.6,'Te':	5.5,'I':	5.35,'Xe':	4.044,'Cs':	59.42,'Ba':	39.7,'La':	31.1,'Ce':	29.6,'Pr':	28.2,'Nd':	31.4,'Pm':	30.1,
    'Sm':	28.8,'Eu':	27.7,'Gd':	23.5,'Tb':	25.5,'Dy':	24.5,'Ho':	23.6,'Er':	22.7,'Tm':	21.8,'Yb':	21,'Lu':	21.9,'Hf':	16.2,'Ta':	13.1,'W':	11.1,'Re':	9.7,'Os':	8.5,
    'Ir':	7.6,'Pt':	6.5,'Au':	5.8,'Hg':	5.02,'Tl':	7.6,'Pb':	6.8,'Bi':	7.4,'Po':	6.8,'At':	6,'Rn':	5.3,'Fr':	47.1,'Ra':	38.3,'Ac':	32.1,'Th':	32.1,'Pa':	25.4,'U':	24.9,'Np':	24.8,
    'Pu':	24.5,'Am':	23.3,'Cm':	23,'Bk':	22.7,'Cf':	20.5,'Es':	19.7,'Fm':	23.8,'Md':	18.2,'No':	17.5,'Lr':	"n.a."
    }
specific_heat = {
    'H':	14.304,'He':	5.193,'Li':	3.582,'Be':	1.825,'B':	1.026,'C':	0.709,'N':	1.04,'O':	0.918,'F':	0.824,'Ne':	1.03,'Na':	1.228,'Mg':	1.023,'Al':	0.897,'Si':	0.705,'P':	0.769,
    'S':	0.71,'Cl':	0.479,'Ar':	0.52,'K':	0.757,'Ca':	0.647,'Sc':	0.568,'Ti':	0.523,'V':	0.489,'Cr':	0.449,'Mn':	0.479,'Fe':	0.449,'Co':	0.421,'Ni':	0.444,'Cu':	0.385,'Zn':	0.388,
    'Ga':	0.371,'Ge':	0.32,'As':	0.329,'Se':	0.321,'Br':	0.474,'Kr':	0.248,'Rb':	0.363,'Sr':	0.301,'Y':	0.298,'Zr':	0.278,'Nb':	0.265,'Mo':	0.251,'Tc':	0.063,'Ru':	0.238,'Rh':	0.243,
    'Pd':	0.244,'Ag':	0.235,'Cd':	0.232,'In':	0.233,'Sn':	0.228,'Sb':	0.207,'Te':	0.202,'I':	0.214,'Xe':	0.158,'Cs':	0.242,'Ba':	0.204,'La':	0.195,'Ce':	0.192,'Pr':	0.193,'Nd':	0.19,
    'Pm':	0.18,'Sm':	0.197,'Eu':	0.182,'Gd':	0.236,'Tb':	0.182,'Dy':	0.17,'Ho':	0.165,'Er':	0.168,'Tm':	0.16,'Yb':	0.155,'Lu':	0.154,'Hf':	0.144,'Ta':	0.14,'W':	0.132,'Re':	0.137,
    'Os':	0.13,'Ir':	0.131,'Pt':	0.133,'Au':	0.129,'Hg':	0.14,'Tl':	0.129,'Pb':	0.129,'Bi':	0.122,'Po':	0.12,'At':	"n.a",'Rn':	0.094,'Fr':	"n.a",'Ra':	0.092,'Ac':	0.12,'Th':	0.113,
    'Pa':	0.0991,'U':	0.116,'Np':	0.12,'Pu':	0.13,'Am':	0.12,'Cm':	0.13,'Bk':	0.13,'Cf':	0.13,'Es':	0.13,'Fm':	"n.a",'Md':	"n.a",'No':	"n.a",'Lr':	"n.a"
    }
thermal_conductivity = {
    'H':	0.1805,'He':	0.1513,'Li':	85,'Be':	190,'B':	27,'C':	140,'N':	0.02583,'O':	0.02658,'F':	0.0277,'Ne':	0.0491,'Na':	140,'Mg':	160,'Al':	235,'Si':	150,'P':	0.236,
    'S':	0.205,'Cl':	0.0089,'Ar':	0.01772,'K':	100,'Ca':	200,'Sc':	16,'Ti':	22,'V':	31,'Cr':	94,'Mn':	7.8,'Fe':	80,'Co':	100,'Ni':	91,'Cu':	400,'Zn':	120,'Ga':	29,'Ge':	60,
    'As':	50,'Se':	2.04,'Br':	0.12,'Kr':	0.00943,'Rb':	58,'Sr':	35,'Y':	17,'Zr':	23,'Nb':	54,'Mo':	139,'Tc':	51,'Ru':	120,'Rh':	150,'Pd':	72,'Ag':	430,'Cd':	97,'In':	82,
    'Sn':	67,'Sb':	24,'Te':	3,'I':	0.449,'Xe':	0.00565,'Cs':	36,'Ba':	18,'La':	13,'Ce':	11,'Pr':	13,'Nd':	17,'Pm':	17.9,'Sm':	13,'Eu':	14,'Gd':	11,'Tb':	11,'Dy':	11,
    'Ho':	16,'Er':	15,'Tm':	17,'Yb':	39,'Lu':	16,'Hf':	23,'Ta':	57,'W':	170,'Re':	48,'Os':	88,'Ir':	150,'Pt':	72,'Au':	320,'Hg':	8.3,'Tl':	46,'Pb':	35,'Bi':	8,'Po':	20,
    'At':	2,'Rn':	0.00361,'Fr':	77,'Ra':	19,'Ac':	12,'Th':	54,'Pa':	47,'U':	27,'Np':	6,'Pu':	6,'Am':	10,'Cm':	8.8,'Bk':	10,'Cf':	10,'Es':	10,'Fm':	10,'Md':	10,'No':	10,
    'Lr':	10
    }

# Example Crystal Structure from the Materials Project
with MPRester("1p9DyF1cbq2mj1dKRKwQbecYMS3CUjqd") as mpr:
    docs = mpr.materials.summary.search(
    material_ids = 'mp-10144',
    fields=['material_id', 'formula_pretty', 'nsites', 'structure', 'energy_per_atom',
            'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'symmetry'])

crystal_struc = docs[0].structure
adsorbate_specie = 'H'
adsorbate = Molecule(adsorbate_specie, [[0, 0, 0]])

def get_edge_features_and_geometry(structure):
  sga = SpacegroupAnalyzer(structure)
  struc_ = sga.get_conventional_standard_structure()
  asf = AdsorbateSiteFinder(struc_)
  sites = asf.find_adsorption_sites()
  priority = ['hollow', 'bridge', 'ontop']

  def select_array(data, priority):
      for key in priority:
          if key in data and data[key]:
              if isinstance(data[key], list):
                  return random.choice(data[key])
              else:
                  return data[key]
      return None
  site = select_array(sites, priority)
  struc_ = asf.add_adsorbate(adsorbate, site)
  sga = SpacegroupAnalyzer(struc_)
  conv = sga.get_conventional_standard_structure()
  distance = conv.distance_matrix

  atoms = Atoms(conv.labels, positions=(conv.cart_coords), cell=(conv.lattice.abc + conv.lattice.angles), pbc=False)
  nl = NeighborList(neighborlist.natural_cutoffs(atoms), self_interaction=False, bothways=True)
  nl.update(atoms)
  adj_matrix=(nl.get_connectivity_matrix().toarray())

  data1=tf.where(tf.not_equal(adj_matrix, 0)).numpy()
  data3 = []
  for j in range(data1.shape[0]):
    data3.append(distance[list(data1[j,:])[0],list(data1[j,:])[1]])
  min_edge, max_edge = np.array(0.0), np.array(5.5)
  x=((data3-min_edge)/(max_edge-min_edge))*10
  x=(np.array(np.where(x==10, 9, x))).astype(int)
  data5 = []
  for value in x:
    idx = [0 for _ in range(10)]
    idx[value]=1
    data5.append(idx)
  coor = data1
  edge_dist_discrete = data5

  angle = []
  for j in range(len(coor)):
    pos1 = atoms.positions[coor[j, 0]].reshape(1, 3)
    pos2 = atoms.positions[coor[j, 1]].reshape(1, 3)
    if np.sum(np.abs(pos1)) == 0 or np.sum(np.abs(pos2)) == 0:
      angle.append(np.array([0.0]))
    else:
      angle.append(get_angles(pos1, pos2))
  min_angle, max_angle = np.array(0.0), np.array(120.1)
  x=(((angle-min_angle)/(max_angle-min_angle))*10).reshape(len(angle),)
  x=(np.array(np.where(x==10, 9, x))).astype(int)
  data5 = []
  for value in x:
    idx = [0 for _ in range(10)]
    idx[value]=1
    data5.append(idx)
  edge_ang_discrete = data5

  diff = []
  for j in range(len(coor)):
    x1 = electronegativity[conv.labels[coor[j, 0]]]
    x2 = electronegativity[conv.labels[coor[j, 1]]]
    diff.append(np.abs(x1-x2))
  min_x, max_x = np.array(0.0), np.array(2.7)
  diff = [min(idx, max_x) for idx in diff]
  x=(((diff-min_x)/(max_x-min_x))*10).reshape(len(diff),)
  x=(np.array(np.where(x==10, 9, x))).astype(int)
  data5 = []
  for value in x:
    idx = [0 for _ in range(10)]
    idx[value]=1
    data5.append(idx)
  x_diff_discrete=data5

  atoms = Atoms(conv.labels, positions=(conv.cart_coords)*1.8897259886,
              cell=(tuple(np.array(conv.lattice.abc)*1.8897259886) + conv.lattice.angles),
              pbc=False)
  cm= CoulombMatrix(n_atoms_max=atoms.positions.shape[0], permutation='none')
  cm_out = (cm.create(atoms).reshape(atoms.positions.shape[0], atoms.positions.shape[0]))
  esm= EwaldSumMatrix(n_atoms_max=atoms.positions.shape[0], permutation='none')
  esm_out = (esm.create(atoms).reshape(atoms.positions.shape[0], atoms.positions.shape[0]))
  sine= SineMatrix(n_atoms_max=atoms.positions.shape[0], permutation='none')
  sine_out = (sine.create(atoms).reshape(atoms.positions.shape[0], atoms.positions.shape[0]))
  data1 = []
  for j in range(len(coor)):
    data1.append(cm_out[tuple(coor[j])])
  cm_data = data1
  data1 = []
  for j in range(len(coor)):
    data1.append(esm_out[tuple(coor[j])])
  esm_data = data1
  data1 = []
  for j in range(len(coor)):
    data1.append(sine_out[tuple(coor[j])])
  sine_data = data1
  min_cm = np.array(cm_data).min()
  max_cm = np.array(cm_data).max()
  for j in range(len(cm_data)):
    if cm_data[j] == float('inf'):
      new = sorted(set(cm_data), reverse=True)[1]
      cm_data = [new if item == max_cm else item for item in cm_data]
  min_esm = np.array(esm_data).min()
  max_esm = np.array(esm_data).max()
  for j in range(len(esm_data)):
    if esm_data[j] == float('inf'):
      new = sorted(set(esm_data), reverse=True)[1]
      esm_data = [new if item == max_esm else item for item in esm_data]
  min_sine = np.array(sine_data).min()
  max_sine = np.array(sine_data).max()
  for j in range(len(sine_data)):
    if sine_data[j] == float('inf'):
      new = sorted(set(sine_data), reverse=True)[1]
      sine_data = [new if item == max_sine else item for item in sine_data]
  min_cm, max_cm = np.array(0.439), np.array(53200.0)
  k =  10/(np.log10(max_cm)-np.log10(min_cm))
  C = -1*np.log10(min_cm)*k
  log_cm = (k*np.log10(np.array(cm_data)) + C).astype(int)
  log_cm = (np.array(np.where(log_cm==10, 9, log_cm))).astype(int)
  temp = []
  for value in log_cm:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    temp.append(idx)
  log_cm_discrete= temp
  min_sine, max_sine = np.array(0.232), np.array(19107210.0)
  k =  10/(np.log10(max_sine)-np.log10(min_sine))
  C = -1*np.log10(min_sine)*k
  log_sine = (k*np.log10(np.array(sine_data)) + C).astype(int)
  log_sine = (np.array(np.where(log_sine==10, 9, log_sine))).astype(int)
  temp = []
  for value in log_sine:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    temp.append(idx)
  log_sine_discrete = temp
  min_asinh_esm, max_asinh_esm = np.array(-3.2), np.array(5.4)
  asinh_esm = np.log10(np.array(esm_data) + np.sqrt((np.array(esm_data)**2) + 1))
  asinh_esm = (((np.array(asinh_esm)-min_asinh_esm)/(max_asinh_esm-min_asinh_esm))*10).astype(int)
  esm_norm = ((np.array(np.where(asinh_esm==10, 9, asinh_esm))).astype(int))
  temp = []
  for value in esm_norm:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    temp.append(idx)
  asinh_esm_discrete = temp
  temp = []
  for j in range(len(coor)):
    temp.append(edge_dist_discrete[j]+x_diff_discrete[j]+edge_ang_discrete[j]+log_sine_discrete[j]+asinh_esm_discrete[j]+log_sine_discrete[j])
  edge_feats = temp

  return (coor, edge_feats, conv, site)

data = get_edge_features_and_geometry(crystal_struc)
coor = data[0]
edge_feats = data[1]
conv_struc = data[2]
site = data[3]

def get_node_features(conv_struc):
  label = conv_struc.labels
  data1 =[]
  for j in range(len(label)):
    xx = (atomic_no[label[j]]-1)
    data1.append(xx)
  data11 = []
  for value in data1:
    idx = [0 for _ in range(100)]
    idx[value] = 1
    data11.append(idx)
  Z = data11

  data2 =[]
  for j in range(len(label)):
    xx = (group_no[label[j]]-1)
    data2.append(xx)
  data22 = []
  for value in data2:
    idx = [0 for _ in range(18)]
    idx[value] = 1
    data22.append(idx)
  GN = data22

  data3 =[]
  for j in range(len(label)):
    xx = (row_no[label[j]]-1)
    data3.append(xx)
  data33 = []
  for value in data3:
    idx = [0 for _ in range(9)]
    idx[value] = 1
    data33.append(idx)
  RN = data33

  data4 =[]
  for j in range(len(label)):
    xx = (valence[label[j]]-1)
    data4.append(xx)
  data44 = []
  for value in data4:
    idx = [0 for _ in range(9)]
    idx[value] = 1
    data44.append(idx)
  VL = data44

  data5 = []
  for j in range(len(label)):
    xx = (block[label[j]]-1)
    data5.append(xx)
  data55 = []
  for value in data5:
    idx = [0 for _ in range(4)]
    idx[value] = 1
    data55.append(idx)
  BK = data55

  data6 =[]
  for j in range(len(label)):
    xx = (electronegativity[label[j]])
    xx=((xx-0)/(3.98-0))*10
    xx=(np.array(np.where(xx==10, 9, xx))).astype(int)
    data6.append(xx)
  data66 = []
  for value in data6:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    data66.append(idx)
  X = data66

  data7 =[]
  for j in range(len(label)):
    xx = (covalent_radius[label[j]])
    xx = ((xx-0.28)/(2.6-0.28))*10
    xx=(np.array(np.where(xx==10, 9, xx))).astype(int)
    data7.append(xx)
  data77 = []
  for value in data7:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    data77.append(idx)
  CR = data77

  data8 = []
  for j in range(len(label)):
    xx = (ionization_energy[label[j]])
    xx = (9.53934*np.log10(xx)-3.26649).astype(int)
    xx=(np.array(np.where(xx==10, 9, xx))).astype(int)
    data8.append(xx)
  data88 = []
  for value in data8:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    data88.append(idx)
  IE = data88

  data9 =[]
  for j in range(len(label)):
    xx = (electron_affinity[label[j]])
    xx = ((xx-(-2.33))/(3.612724-(-2.33)))*10
    xx=(np.array(np.where(xx==10, 9, xx))).astype(int)
    data9.append(xx)
  data99 = []
  for value in data9:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    data99.append(idx)
  EA = data99

  data10 =[]
  for j in range(len(label)):
    xx = (molar_volume[label[j]])
    xx = (8.03842*np.log10(xx)-5.16439).astype(int)
    xx=(np.array(np.where(xx==10, 9, xx))).astype(int)
    data10.append(xx)
  data100 = []
  for value in data10:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    data100.append(idx)
  MV = data100

  data11 =[]
  for j in range(len(label)):
    xx = (average_ionic_radius[label[j]])
    xx = ((xx-0)/(1.94-0))*10
    xx=(np.array(np.where(xx==10, 9, xx))).astype(int)
    data11.append(xx)
  data111 = []
  for value in data11:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    data111.append(idx)
  IR = data111

  data12 =[]
  for j in range(len(label)):
    xx = (polarizability[label[j]])
    xx = ((xx-0.204956)/(59.42-0.204956))*10
    xx=(np.array(np.where(xx==10, 9, xx))).astype(int)
    data12.append(xx)
  data122 = []
  for value in data12:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    data122.append(idx)
  PZ = data122

  data13 =[]
  for j in range(len(label)):
    xx = (specific_heat[label[j]])
    xx = (4.24427*np.log10(xx)+5.0959).astype(int)
    xx=(np.array(np.where(xx==10, 9, xx))).astype(int)
    data13.append(xx)
  data133 = []
  for value in data13:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    data133.append(idx)
  SH = data133

  data14 =[]
  for j in range(len(label)):
    xx = (thermal_conductivity[label[j]])
    xx = (1.97007*np.log10(xx)+4.81188).astype(int)
    xx=(np.array(np.where(xx==10, 9, xx))).astype(int)
    data14.append(xx)
  data144 = []
  for value in data14:
    idx = [0 for _ in range(10)]
    idx[value] = 1
    data144.append(idx)
  TC = data144

  prop = []
  for j in range(len(label)):
    prop.append(Z[j]+GN[j]+RN[j]+VL[j]+BK[j]+X[j]+CR[j]+IE[j]+EA[j]+MV[j]+IR[j]+PZ[j]+SH[j]+TC[j])
  node = prop

  return node

node_features = get_node_features(conv_struc)

sga = SpacegroupAnalyzer(crystal_struc)
conv_struc_ = sga.get_conventional_standard_structure()
calc = XRDCalculator(wavelength='CuKa')
pattern = calc.get_pattern(conv_struc_, scaled=True, two_theta_range=(0, 180))
new_y = np.zeros(180)
for j in range(len(pattern.x)):
  new_y[int(pattern.x[j])] = pattern.y[j]
  new_y = np.array(new_y).reshape(-1,1)
xrd = MinMaxScaler().fit_transform(new_y).reshape(1,-1)
xrd = torch.tensor((xrd.reshape(180)), dtype=torch.float).unsqueeze(0)

edges = torch.tensor(np.array(coor).T, dtype=torch.int64)
edge_features =  torch.tensor((edge_feats), dtype=torch.int)
node_features = torch.tensor((node_features), dtype=torch.float)
data = Data(x=node_features, edge_index=edges, edge_attr=edge_features, glob_attr=xrd)

# Saved Pytorch Models can be downloaded from the "Saved Models Folder".
# Learning Models Codes can be found in the "Learning Models" Folder.

model_dos.eval()
with torch.no_grad():
  y_dos = model_dos(data.to(device))
y_dos = y_dos.cpu().numpy()

dos = torch.tensor((y_dos.reshape(1024)), dtype=torch.float).unsqueeze(0)
global_features = torch.cat((dos, xrd), 1)
y_dummy = torch.tensor((np.array([1]).reshape(1)), dtype=torch.float)
data_ads = Data(x=node_features, edge_index=edges, edge_attr=edge_features, glob_attr = global_features, y=y_dummy)

model_ads.eval()
with torch.no_grad():
  y_ads = model_ads(data_ads.to(device))
y_ads = y_ads.cpu().numpy()

print('Adsorbate Energy from validation:', y_ads[0][0],'eV')
