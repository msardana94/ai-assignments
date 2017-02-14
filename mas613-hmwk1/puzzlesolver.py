import re,sys,heapq
"""
The search algorithms are defined below which are same for all the problem statements.
"""

def hash_func(item):
	return str(hash(tuple(item)))

def bfs(obj):
	frontier_list = [(obj.init_state,[obj.init_state])]
	explored_list = {}
	explored_list[hash_func(obj.init_state)] = 1
	num_of_nodes = 1
	max_frontier_len = 1
	while frontier_list:
		if max_frontier_len < len(frontier_list):
			max_frontier_len = len(frontier_list)
		(node,path) = frontier_list.pop(0)
		if obj.goal_test(node):
			return path,num_of_nodes,(max_frontier_len,len(explored_list))
		nodelist = obj.get_states(node)
		num_of_nodes+=len(nodelist)
		for (nextnode,cost) in nodelist:
			if obj.goal_test(nextnode):
				return path+[nextnode],num_of_nodes,(max_frontier_len,len(explored_list))
			if hash_func(nextnode) not in explored_list:
				explored_list[hash_func(nextnode)] = 1
				frontier_list.append((nextnode, path + [nextnode]))
	return None,num_of_nodes,(max_frontier_len,len(explored_list))

def dfs(obj):
	frontier_list = [(obj.init_state,[obj.init_state])]
	num_of_nodes = 1
	max_frontier_len = 1
	while frontier_list:
		if max_frontier_len < len(frontier_list):
			max_frontier_len = len(frontier_list)
		(node,path) = frontier_list.pop()
		if obj.goal_test(node):
			return path,num_of_nodes,(max_frontier_len,None)
		nodelist = obj.get_states(node)
		num_of_nodes+=len(nodelist)

		for (nextnode,cost) in nodelist:
			if obj.goal_test(nextnode):
				return path+[nextnode],num_of_nodes,(max_frontier_len,None)
			print [x[0] for x in frontier_list]
			print nextnode
			if nextnode not in [x[0] for x in frontier_list] and nextnode != obj.init_state:
				frontier_list.append((nextnode, path + [nextnode]))
	return None,num_of_nodes,(max_frontier_len,len(explored_list))

def dldfs(obj,depthLimit):
	frontier_list = [(obj.init_state,[obj.init_state],0)]
	num_of_nodes = 0
	max_frontier_len = 0
	while frontier_list:
		if max_frontier_len < len(frontier_list):
			max_frontier_len = len(frontier_list)
		(node,path,depth) = frontier_list.pop()
		if obj.goal_test(node):
			return path,num_of_nodes,max_frontier_len
		if depth == depthLimit:
			continue
		nodelist = obj.get_states(node)
		num_of_nodes+=len(nodelist)

		for (nextnode,cost) in nodelist:
			if obj.goal_test(nextnode):
				return path+ [nextnode],num_of_nodes,max_frontier_len
			frontier_list.append((nextnode, path + [nextnode],depth+1))

	return None,num_of_nodes,max_frontier_len


def iddfs(obj):
	depth = 0
	num_of_nodes = 0
	max_frontier_len = 0
	while True:
		path,numNodes,maxFrontier = dldfs(obj,depth)
		num_of_nodes+=numNodes
		if max_frontier_len < maxFrontier:
			max_frontier_len = maxFrontier
		if path is not None:
			return path,num_of_nodes,(max_frontier_len,None)
		depth+=1
	return None,num_of_nodes,(max_frontier_len,len(explored_list))

def uniformcost(obj):
	frontier_list = [(0,obj.init_state,[obj.init_state])]
	explored_list = {}
	explored_list[hash_func(obj.init_state)] = 1
	num_of_nodes = 1
	max_frontier_len = 1
	while frontier_list:
		if max_frontier_len < len(frontier_list):
			max_frontier_len = len(frontier_list)
		(cost,node,path) = heapq.heappop(frontier_list)
		if obj.goal_test(node):
			return path,num_of_nodes,(max_frontier_len,len(explored_list))

		hash_val = hash_func(node)
		if hash_val in explored_list and explored_list[hash_val] < cost:
			continue
		nodelist = obj.get_states(node)
		num_of_nodes+=len(nodelist)
		
		for (nextnode,cost) in nodelist:
			if obj.goal_test(nextnode):
				nextnode_cost = 0
			else:
				nextnode_cost = 1
			
			if hash_func(nextnode) not in explored_list:
				heapq.heappush(frontier_list,(cost+nextnode_cost,nextnode, path + [nextnode]))
		explored_list[hash_val] = cost
	
	return None,num_of_nodes,(max_frontier_len,len(explored_list.keys()))

def greedySearch(obj):
	frontier_list = [(obj.get_heuristic(obj.init_state),obj.init_state,[obj.init_state])]
	explored_list = {}
	explored_list[hash_func(obj.init_state)] = 1
	num_of_nodes = 1
	max_frontier_len = 1
	while frontier_list:
		if max_frontier_len < len(frontier_list):
			max_frontier_len = len(frontier_list)
		(cost,node,path) = heapq.heappop(frontier_list)
		if obj.goal_test(node):
			return path,num_of_nodes,(max_frontier_len,len(explored_list))
		nodelist = obj.get_states(node)
		num_of_nodes+=len(nodelist)
		
		for (nextnode,cost) in nodelist:
			hash_val = hash_func(nextnode)
			if hash_val not in explored_list:
				heapq.heappush(frontier_list,(obj.get_heuristic(nextnode),nextnode, path + [nextnode]))
				explored_list[hash_val] = 1
	
	return None,num_of_nodes,(max_frontier_len,len(explored_list))

def astar(obj):
	eval_func = obj.get_heuristic(obj.init_state)
	frontier_list = [(eval_func,0,obj.init_state,[obj.init_state])]
	explored_list = {}
	explored_list[hash_func(obj.init_state)] = 1
	num_of_nodes = 1
	max_frontier_len = 1
	while frontier_list:
		if max_frontier_len < len(frontier_list):
			max_frontier_len = len(frontier_list)
		(eval_func,init_cost,node,path) = heapq.heappop(frontier_list)
		if obj.goal_test(node):
			return path,num_of_nodes,(max_frontier_len,len(explored_list))
		nodelist = obj.get_states(node)
		num_of_nodes+=len(nodelist)

		for (nextnode,cost) in nodelist:
			hash_val = hash_func(nextnode)
			if hash_val not in explored_list:
				nextnode_cost = init_cost+cost
				nextnode_eval_func = obj.get_heuristic(nextnode) + nextnode_cost
				heapq.heappush(frontier_list,(nextnode_eval_func,nextnode_cost,nextnode, path + [nextnode]))
				explored_list[hash_val] = 1
	
	return None,num_of_nodes,(max_frontier_len,len(explored_list))

def costLimitedDFS(obj,cut_off):
	eval_func = obj.get_heuristic(obj.init_state)
	frontier_list = [(eval_func,0,obj.init_state,[obj.init_state])]
	next_min = float("inf")
	num_of_nodes = 0
	max_frontier_len = 0
	while frontier_list:
		if max_frontier_len < len(frontier_list):
			max_frontier_len = len(frontier_list)
		(eval_func,init_cost,node,path) = heapq.heappop(frontier_list)
		if eval_func <= cut_off:
			if obj.goal_test(node):
				return (path,next_min,num_of_nodes,max_frontier_len)
			nodelist = obj.get_states(node)
			num_of_nodes+=len(nodelist)

			for (nextnode,cost) in nodelist:
				nextnode_cost = init_cost+cost
				nextnode_eval_func = obj.get_heuristic(nextnode) + nextnode_cost
				heapq.heappush(frontier_list,(nextnode_eval_func,nextnode_cost,nextnode, path + [nextnode]))
		elif eval_func<next_min:
			next_min = eval_func
	return (None,next_min,num_of_nodes,max_frontier_len)

def idastar(obj):
	cut_off = obj.get_heuristic(obj.init_state)
	num_of_nodes = 1
	max_frontier_len = 1
	while True:
		(path,cut_off,numNodes,maxFrontier) = costLimitedDFS(obj,cut_off)
		num_of_nodes+=numNodes
		if max_frontier_len < maxFrontier:
			max_frontier_len = maxFrontier
		if not path is None:
			return path,num_of_nodes,(max_frontier_len,None)
		if cut_off == float("inf"):
			return None,num_of_nodes,(max_frontier_len,None)

"""
Class for water jug puzzle
"""
class waterJug:
	def __init__(self,init_state,goal_state,jugCapacity,func_name):
		self.numOfJugs = len(jugCapacity)
		self.jugCapacity = jugCapacity
		self.init_state = init_state
		self.goal_state = goal_state
		self.func_name = func_name
		
	def get_states(self,cur_state):
		new_states = []
		jugCapacity = self.jugCapacity
		if self.numOfJugs == 2:
			new_states = [(list(cur_state),1) for i in range(0,6)]
			# if action == "A1":
			new_states[0][0][1] = jugCapacity[1]
			# elif action == "A2":
			new_states[1][0][0] = jugCapacity[0]
			# elif action == "A3":
			new_states[2][0][1] = 0
			# elif action == "A4":
			new_states[3][0][0] = 0
			# elif action == "A5":
			d = min(jugCapacity[1] - cur_state[1],cur_state[0])
			new_states[4][0][0] = new_states[4][0][0]-d
			new_states[4][0][1] = new_states[4][0][1]+d
			# elif action == "A6":
			d = min(jugCapacity[0] - cur_state[0],cur_state[1])
			new_states[5][0][0] = new_states[5][0][0]+d
			new_states[5][0][1] = new_states[5][0][1]-d
		return new_states

	def goal_test(self,cur_state):
		return cur_state == self.goal_state

	def get_heuristic(self,cur_state):
		goal_state = self.goal_state
		if self.func_name == "absolute_difference":
			jugCapacity = self.jugCapacity
			if self.numOfJugs == 2:
				if goal_state[0] != 0:
					return abs(goal_state[0]-jugCapacity[0])
				else:
					return abs(goal_state[1]-jugCapacity[1])
			elif self.numOfJugs == 3:
				if goal_state[0] != 0:
					return abs(goal_state[0]-jugCapacity[0])
				elif goal_state[1] != 0:
					return abs(goal_state[1]-jugCapacity[1])
				else:
					return abs(goal_state[2]-jugCapacity[2])
		elif self.func_name == "goal_test":
			if cur_state == goal_state:
				return 0
			else:
				return 1


"""
Class for Path planning puzzle
"""
class pathPlanning():
	def __init__(self,graph,location,init_state,goal_state,func_name):
		self.graph = graph
		self.location = location
		self.init_state = init_state
		self.goal_state = goal_state
		self.func_name = func_name

	def createGraph(self,contents):
		for line in contents:
			if line:
				arr = re.match(r'\((.+)\)',line).group(1).split(',')
				arr[0] = arr[0].replace('"','').strip(' ')
				arr[1] = arr[1].replace('"','').strip(' ')
				arr[2] = int(arr[2].strip(' '))
				self.graph[arr[0]].append((arr[1],arr[2]))
				self.graph[arr[1]].append((arr[0],arr[2]))

	def get_states(self,cur_state):
		return self.graph[cur_state]

	def goal_test(self,cur_state):
		return cur_state == self.goal_state

	def get_heuristic(self,cur_state):
		goal_state = self.goal_state
		location = self.location

		if self.func_name == "manhattan_dist":
			return abs(location[cur_state][0] - location[goal_state][0]) + abs(location[cur_state][1] - location[goal_state][1])
		elif self.func_name == "euclidean_dist":
			return int(pow((location[cur_state][0] - location[goal_state][0]) **2 + abs(location[cur_state][1] - location[goal_state][1]) **2 , 0.5))
		elif self.func_name == "goal_test":
			if cur_state == goal_state:
				return 0
			else:
				return 1

"""
class for Burnt Pancake puzzle
"""
class burntPancake():
	def __init__(self,init_state,func_name,maxnum):
		self.init_state = init_state
		self.goal_state = range(1,maxnum)
		self.maxnum = maxnum
		self.func_name = func_name


	def get_states(self,cur_state):
		new_states = [(list(cur_state),1) for i in range(1,self.maxnum)]
		new_states[0][0][0] = -new_states[0][0][0]
		for i in range(2,self.maxnum):
			for j in range(0,i/2):
				temp = -new_states[i-1][0][i-j-1]
				new_states[i-1][0][i-j-1] = -new_states[i-1][0][j]
				new_states[i-1][0][j] = temp
		return new_states

	def goal_test(self,cur_state):
		return cur_state == self.goal_state

	def get_heuristic(self,cur_state):
		goal_state = self.goal_state
		if self.func_name == "misplaced_num":
			count=0
			for i in range(1,self.maxnum):
				if abs(cur_state[i-1])!=i:
					count+=1
			return count
		
		elif self.func_name == "goal_test":
			if cur_state == goal_state:
				return 0
			else:
				return 1

def driver_function(filename,searchAlgo,heuristic_func="goal_test"):
	with open(filename) as f:
		lines = f.readlines()
		lines = [line.strip('\r\n') for line in lines]
		path = []
		obj = None
		time_complexity = 0
		space_complexity = (0,0)
		if heuristic_func not in ["goal_test","misplaced_num","manhattan_dist","euclidean_dist","absolute_difference"]:
			print "Incorrect keyword for heuristic function"
			return
		if lines[0] == "jugs":
			capacity = re.match(r'\((.+)\)',lines[1]).group(1).split(',')
			capacity = [int(val.strip(' ')) for val in capacity]

			init_state = re.match(r'\((.+)\)',lines[2]).group(1).split(',')
			init_state = [int(val.strip(' ')) for val in init_state]

			goal_state = re.match(r'\((.+)\)',lines[3]).group(1).split(',')
			goal_state = [int(val.strip(' ')) for val in goal_state]

			# print capacity,init_state,goal_state
			obj = waterJug(init_state,goal_state,capacity,heuristic_func)

		elif lines[0] == "cities":
			cityinfo = re.findall(r'\((.*?)\)+',lines[1])
			graph = {}
			location = {}
			for city in cityinfo:
				arr = city.split(',')
				# print arr
				arr[0] = arr[0].replace('"','')
				graph[arr[0]] = []
				location[arr[0]] = (int(arr[1].strip(' ')),int(arr[2].strip(' ')))
			obj = pathPlanning(graph,location,lines[2].replace('"','').strip(' '), lines[3].replace('"','').strip(' '),heuristic_func)
			obj.createGraph(lines[4:])

		elif lines[0] == "pancakes":
			init_state = re.findall(r'\((.+)\)',lines[1])[0].split(',')
			init_state = [int(val.strip(' ')) for val in init_state]
			maxnum = 0
			for i in init_state:
				if abs(i)>maxnum:
					maxnum = abs(i)

			obj = burntPancake(init_state,heuristic_func,maxnum+1)
			# print obj.goal_state
		
		if searchAlgo == "bfs":
			path,time_complexity,space_complexity = bfs(obj)
			# cost = len(path)
		elif searchAlgo == "dfs":
			path,time_complexity,space_complexity = dfs(obj)
			# cost = len(path)
		elif searchAlgo == "unicost":
			path,time_complexity,space_complexity = uniformcost(obj)
		elif searchAlgo == "iddfs":
			path,time_complexity,space_complexity = iddfs(obj)
		elif searchAlgo == "greedy":
			path,time_complexity,space_complexity = greedySearch(obj)
		elif searchAlgo == "astar":
			path,time_complexity,space_complexity = astar(obj)
		elif searchAlgo == "idastar":
			path,time_complexity,space_complexity = idastar(obj)
			# cost = len(path)

		print "Time Complexity:"+str(time_complexity)
		print "Space Complexity - Frontier List size:"+str(space_complexity[0])
		if not space_complexity[1] is None:
			print "Space Complexity - Explored List size:"+str(space_complexity[1])

		if path:
			# print "Path cost:"+str(cost)
			for state in path:
				print state
		else:
			print "No Solution\n"

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print "Insufficient arguments. Follow the command: python puzzlesolver.py function_name algo_name heuristic_func(optional) "
	if len(sys.argv) == 3:
		driver_function(sys.argv[1],sys.argv[2])
	if len(sys.argv) == 4:
		driver_function(sys.argv[1],sys.argv[2],sys.argv[3])

	