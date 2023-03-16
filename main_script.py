import decision_tree

#read second data file 'ks_vs_kr_data.txt' and create a C4.5 decision tree
c45_dataset1 = decision_tree.C45()
with open('kr_vs_kp_name.txt', 'r') as file:
	classes = file.readline()
	c45_dataset1.classes = [x.strip() for x in classes.split(',')]
	# add attributes
	for line in file:
		[attribute, values] = [x.strip() for x in line.split(':')]
		values = [x.strip() for x in values.split(',')]
		c45_dataset1.attrValues[attribute] = values
c45_dataset1.numAttributes = len(c45_dataset1.attrValues.keys())
c45_dataset1.attributes = list(c45_dataset1.attrValues.keys())
with open('kr_vs_kp_data.txt', 'r') as file:
	for line in file:
		row = [x.strip() for x in line.split(',')]
		if row != [] or row != [""]:
			c45_dataset1.data.append(row)
#preprocess data for numeric features
c45_dataset1.preprocessData()
#Train a c4.5 decision tree
c45_dataset1.generateTree()
#visualize the trained decision tree
c45_dataset1.plotTree('kr_vs_kp')

#read second data file 'heart_data.txt' and create a C4.5 decision tree
c45_dataset2 = decision_tree.C45()
with open('heart_name.txt', 'r') as file:
	classes = file.readline()
	c45_dataset2.classes = [x.strip() for x in classes.split(',')]
	# add attributes
	for line in file:
		[attribute, values] = [x.strip() for x in line.split(':')]
		values = [x.strip() for x in values.split(',')]
		c45_dataset2.attrValues[attribute] = values
c45_dataset2.numAttributes = len(c45_dataset2.attrValues.keys())
c45_dataset2.attributes = list(c45_dataset2.attrValues.keys())
with open('heart_data.txt', 'r') as file:
	for line in file:
		row = [x.strip() for x in line.split(' ')]
		if row != [] or row != [","]:
			c45_dataset2.data.append(row)
#preprocess data for numeric features
c45_dataset2.preprocessData()
#Train a c4.5 decision tree
c45_dataset2.generateTree()
#visualize the trained decision tree
c45_dataset2.plotTree('heart_data_c45')