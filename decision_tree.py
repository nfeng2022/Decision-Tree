import math
import pydot
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'E:/Program Files (x86)/Graphviz2.38/bin' #add path where you install graphviz


class C45:
	# construct a c4.5 decision tree class for furture use
	def __init__(self):
		self.data = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.target_name = 'class'
		self.tree = None

	def preprocessData(self):
		# do this if data file includes continuous features to turn values into floating number
		for index,row in enumerate(self.data):
			for attr_index in range(self.numAttributes):
				if(not self.isAttrDiscrete(self.attributes[attr_index])):
					self.data[index][attr_index] = float(self.data[index][attr_index])

	def plotTree(self, name):
		#plot tree recursively using pydot
		graph1 = pydot.Dot(graph_type='digraph')
		graph1 = self.plotNode(self.tree, graph1)
		graph1.write_png(name+'.png')

	def plotNode(self, node, graph):
		#plot tree node recursively
		if not node.isLeaf:
			if node.threshold is None:
				#discrete
				for index, child in enumerate(node.children):
					if child.label != 'Fail':
						edge = pydot.Edge(node.label, child.label, label=self.attrValues[node.label][index])
						graph.add_edge(edge)
						self.plotNode(child, graph)
			else:
				#numerical
				leftChild = node.children[0]
				rightChild = node.children[1]
				edge = pydot.Edge(node.label, leftChild.label, label=' <= ' + str(node.threshold))
				graph.add_edge(edge)
				self.plotNode(leftChild, graph)
				edge = pydot.Edge(node.label, rightChild.label, label=' > ' + str(node.threshold))
				graph.add_edge(edge)
				self.plotNode(rightChild, graph)
		return graph



	def generateTree(self):
		#generate a c4.5 tree recursively
		self.tree = self.recursiveGenerateTree(self.data, self.attributes)

	remainingAttributes = []
	def recursiveGenerateTree(self, curData, curAttributes):
		global remainingAttributes
		if len(curData) == 0:
			# Fail
			return Node(True, "Fail", None)
		allSame = self.allSameClass(curData)
		if allSame is not False:
			# return a node with that class
			return Node(True, 'Class: ' + allSame, None)
		elif len(curAttributes) == 0:
			# return a node with the majority class
			majClass = self.getMajClass(curData)
			return Node(True, 'Class: ' + majClass, None)
		else:
			(best, best_threshold, splitted) = self.splitAttribute(curData, curAttributes)
			remainingAttributes = curAttributes[:]
			remainingAttributes.remove(best)
			node = Node(False, best, best_threshold)
			for subset in splitted:
				node.children.append(self.recursiveGenerateTree(subset, remainingAttributes))
			return node

	def getMajClass(self, curData):
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]

	def allSameClass(self, data):
		for row in data:
			if row[-1] != data[0][-1]:
				return False
		return data[0][-1]

	def isAttrDiscrete(self, attribute):
		if attribute not in self.attributes:
			raise ValueError('Attribute not listed')
		elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == 'continuous':
			return False
		else:
			return True

	def splitAttribute(self, curData, curAttributes):
		splitted = []
		maxEnt = -1*float('inf')
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_threshold = None
		for attribute in curAttributes:
			indexOfAttribute = self.attributes.index(attribute)
			if self.isAttrDiscrete(attribute):
				#split curData into n-subsets, where n is the number of 
				#different values of attribute i. Choose the attribute with
				#the max gain
				valuesForAttribute = self.attrValues[attribute]
				subsets = [[] for a in valuesForAttribute]
				for row in curData:
					for index in range(len(valuesForAttribute)):
						if row[indexOfAttribute] == valuesForAttribute[index]:
							subsets[index].append(row)
							break
				e = self.gain(curData, subsets)
				if e > maxEnt:
					maxEnt = e
					splitted = subsets
					best_attribute = attribute
					best_threshold = None
			else:
				#sort the data according to the column.Then try all 
				#possible adjacent pairs. Choose the one that 
				#yields maximum gain
				curData.sort(key = lambda x: x[indexOfAttribute])
				for j in range(0, len(curData) - 1):
					if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
						threshold = (curData[j][indexOfAttribute] + curData[j+1][indexOfAttribute]) / 2
						less = []
						greater = []
						for row in curData:
							if row[indexOfAttribute] > threshold:
								greater.append(row)
							else:
								less.append(row)
						e = self.gain(curData, [less, greater])
						if e >= maxEnt:
							splitted = [less, greater]
							maxEnt = e
							best_attribute = attribute
							best_threshold = threshold
		return (best_attribute,best_threshold,splitted)

	def gain(self, unionSet, subsets):
		#input : data and disjoint subsets of it
		#output : information gain
		S = len(unionSet)
		#calculate impurity before split
		impurityBeforeSplit = self.entropy(unionSet)
		#calculate impurity after split
		weights = [len(subset)/S for subset in subsets]
		impurityAfterSplit = 0
		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.entropy(subsets[i])
		#calculate total gain
		totalGain = impurityBeforeSplit - impurityAfterSplit
		return totalGain

	def entropy(self, dataSet):
		S = len(dataSet)
		if S == 0:
			return 0
		num_classes = [0 for i in self.classes]
		for row in dataSet:
			classIndex = list(self.classes).index(row[-1])
			num_classes[classIndex] += 1
		num_classes = [x/S for x in num_classes]
		ent = 0
		for num in num_classes:
			ent += num*self.log(num)
		return ent*-1

	def log(self, x):
		if x == 0:
			return 0
		else:
			return math.log(x, 2)


class Node:
	def __init__(self, isLeaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.children = []
