import numpy as np

VOVELS     = 'eEyYuUiIoOaAęĘóÓąĄ'
CONSONANTS = 'qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM'
DIACRITICS = 'ęĘóÓąĄśŚłŁżŻźŹćĆńŃ'
SPECIALS   = '`~@#$%^&*()-_=+[{]}\\|";:,<.>/?\''

def extractFeatures(example):
	x1 = sum(1 for c in example if c.isupper())
	x2 = sum(1 for c in example if c.islower())
	x3 = sum(1 for c in example if c.isspace())
	x4 = sum(1 for c in example if c.isdigit())
	x5 = sum(1 for c in example if c in VOVELS)
	x6 = sum(1 for c in example if c in CONSONANTS)
	x7 = sum(1 for c in example if c in DIACRITICS)
	x8 = sum(1 for c in example if c in SPECIALS)
	return np.array([x1,x2,x3,x4,x5,x6,x7,x8])
	
# 3M Poland,		0
# 2,5,1,1,2,5,0,0,	0
