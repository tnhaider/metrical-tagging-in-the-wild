import sys
from collections import Counter

def get_lines(infile):
	countdict = {}
	f = open(infile, 'r').readlines()
	for line in f:
		line = line.split('\t')
		#print(line)
		try:
			token = line[1]
			met = line[2]
			pos = line[6]
			syllin = line[8]
		except IndexError:
			continue
		if syllin == '0':
			countdict.setdefault(pos, []).append(met)
	return countdict

cdict = get_lines(sys.argv[1])
for pos, mets in cdict.items():
	cmets = Counter(mets)
	#print(pos, cmets)
	plus = float(cmets['+'])
	minus = float(cmets['-'])
	if minus == 0:
		minus = 0.0001
	ratio = plus/(plus + minus)
	rratio = plus/minus
	print(pos, cmets, round(rratio, 3), round(ratio, 3))
