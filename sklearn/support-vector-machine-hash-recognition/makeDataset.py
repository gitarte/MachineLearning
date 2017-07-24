import extractFeatures as ef
import numpy as np

VOVELS     = 'eEyYuUiIoOaAęĘóÓąĄ'
CONSONANTS = 'qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM'
DIACRITICS = 'ęĘóÓąĄśŚłŁżŻźŹćĆńŃ'
SPECIALS   = '`~@#$%^&*()-_=+[{]}\\|";:,<.>/?\''

with open('./input.csv','r',encoding='utf-8') as input_file:
	with open('./dataset.csv','w',encoding='utf-8') as dataset:
		for line in input_file:
			r       = line.split(',')
			x       = r[0].lstrip().rstrip()
			y       = r[1].lstrip().rstrip()
			
			example = ef.extractFeatures(x)
			result  = '{0},{1}\n'.format(
				np.array2string(example, separator=','),
				y
			)
			result = result.replace('[','')
			result = result.replace(']','')
			result = result.replace(' ','')
			dataset.write(result)
