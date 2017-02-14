import ast,sys

def minimax(state):
	v = maxValue(state)
	print "\n"
	print v

def minValue(state):
	sys.stdout.write(state[0]+" ")
	if type(state) is tuple:
		return state[1]
	v = sys.maxint
	for s in state[1:]:
		v  = min(v,maxValue(s))
	return v

def maxValue(state):
	sys.stdout.write(state[0]+" ")
	if type(state) is tuple:
		return state[1]
	v = -sys.maxint - 1
	for s in state[1:]:
		v  = max(v,minValue(s))
	return v

def minimax_pruning(state):
	alpha = -sys.maxint-1
	beta = sys.maxint
	v = maxValue_pruning(state,alpha,beta)
	print "\n"
	print v

def minValue_pruning(state,alpha,beta):
	sys.stdout.write(state[0]+" ")
	# print "alpha=",alpha,"beta=",beta
	if type(state) is tuple:
		return state[1]
	v = sys.maxint
	# print state[1:]
	for s in state[1:]:
		# print "min",s,alpha,beta
		v  = min(v,maxValue_pruning(s,alpha,beta))
		# print "v=",v,"beta=",beta
		if v<=alpha:
			# print " v<=alpha return"
			return v
		
		beta = min(beta,v)
	return v

def maxValue_pruning(state,alpha,beta):
	sys.stdout.write(state[0]+" ")
	# print alpha,beta
	if type(state) is tuple:
		# sys.stdout.write(state[0]+" ")
		return state[1]
	v = -sys.maxint - 1
	# print state[1:]
	for s in state[1:]:
		v  = max(v,minValue_pruning(s,alpha,beta))
		if v>=beta:
			# print "v>=beta return"
			return v
		alpha = max(alpha,v)
	return v

def main(filename):
	f = open(filename)
	line = f.readline()
	gametree = ast.literal_eval(line)
	print gametree
	minimax(gametree)
	print "Using Alpha Beta Pruning:"
	minimax_pruning(gametree)
	f.close()

if __name__ == '__main__':
	if len(sys.argv) ==2:
		main(sys.argv[1])
	else:
		print "Insufficient arguments to program!"
