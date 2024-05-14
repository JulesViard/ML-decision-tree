
# # ** ML Algo Classifier :** Decision Trees


# **Subject:** Implementing a Decision Tree algorithm from scratch to predict the position on a user in one of four rooms according to his detection of wifi power of the 7 emiters.


# ## ***Table of Contents***
# - [1. Import dependancies](##step-1---import-dependencies)
# - [2. Loading data](##step-2---loading-data)
# - [3. Creating Decision Tree](##step-3---creating-decision-trees)
# - [4. Evaluation](##step-4---evaluation)
# - [5. Execution](##step-5---execution)
# - [Bonus. Tree Visualisation](##Bonus---Tree-Visualisation)


# ## **Step 1 - import dependencies**


# Import all libraries required

import numpy as np
import matplotlib.pyplot as plt
# import csv


# ## **Step 2 - Loading data**


# We first need to explore a bit our dataset, in order to fully understand our environment.


# #### ***A - Import the data_sets***


# First of all, let's import both of the clean and noisy data_sets


# Import the two datasets

print("Importing the data...")
print("\n")

# Import the clean data_set
clean_data_path = "wifi_db/clean_dataset.txt"
clean_data = np.loadtxt(clean_data_path)

# Import the clean data_set
noisy_data_path = "wifi_db/noisy_dataset.txt"
noisy_data = np.loadtxt(noisy_data_path)


# Here are the numpy array of the two data_sets:


# # Print the data_sets

# print("Clean data_set :\n", clean_data)
# print("\n")
# print("Noisy data_set :\n", noisy_data)


# Let's divide the dataset into data intput and labels


# Split data_sets into y_labels and x_data

clean_data_x = clean_data[:, :-1]
clean_data_y = clean_data[:,-1]

noisy_data_x = noisy_data[:, :-1]
noisy_data_y = noisy_data[:,-1]


# #### ***B - Structure the data_sets***


# Then, in order to have a good idea of the data, let's structure the data_sets by giving significative name to attributes and to labels


# # Structuring the clean data_set

# # # Extract the features (columns 0 to 6) and the target (column 7)
# # features = clean_data[:, 0:7]
# # target = clean_data[:, 7]

# # # Create a structured NumPy array with appropriate field names
# # dtype = [('Emitter 1', float), ('Emitter 2', float), ('Emitter 3', float), ('Emitter 4', float),
# #          ('Emitter 5', float), ('Emitter 6', float), ('Emitter 7', float), ('Room Number', int)]
# # data = np.empty(clean_data.shape[0], dtype=dtype)

# # # Assign the values to the structured array
# # data['Emitter 1'] = features[:, 0]
# # data['Emitter 2'] = features[:, 1]
# # data['Emitter 3'] = features[:, 2]
# # data['Emitter 4'] = features[:, 3]
# # data['Emitter 5'] = features[:, 4]
# # data['Emitter 6'] = features[:, 5]
# # data['Emitter 7'] = features[:, 6]
# # data['Room Number'] = target

# # # Print the structured array in a tabular format
# # print("Clean data_set:\n")
# # header = '{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format(*data.dtype.names)
# # print(header)
# # for row in data:
# #     print('{:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12}'.format(*row))


# # Structuring the noisy data_set

# # # Extract the features (columns 0 to 6) and the target (column 7)
# # features = noisy_data[:, 0:7]
# # target = noisy_data[:, 7]

# # # Create a structured NumPy array with appropriate field names
# # dtype = [('Emitter 1', float), ('Emitter 2', float), ('Emitter 3', float), ('Emitter 4', float),
# #          ('Emitter 5', float), ('Emitter 6', float), ('Emitter 7', float), ('Room Number', int)]
# # data = np.empty(noisy_data.shape[0], dtype=dtype)

# # # Assign the values to the structured array
# # data['Emitter 1'] = features[:, 0]
# # data['Emitter 2'] = features[:, 1]
# # data['Emitter 3'] = features[:, 2]
# # data['Emitter 4'] = features[:, 3]
# # data['Emitter 5'] = features[:, 4]
# # data['Emitter 6'] = features[:, 5]
# # data['Emitter 7'] = features[:, 6]
# # data['Room Number'] = target

# # # Print the structured array in a tabular format
# # print("Noisy data_set:\n")
# # header = '{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format(*data.dtype.names)
# # print(header)
# # for row in data:
# #     print('{:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12}'.format(*row))


# # #### ***C - Some statistics information***


# # We will now take a glance of the data repartition of each attributes and labels


# # Analyse the data - clean data_set

# # Create subplots for histograms
# fig, axes = plt.subplots(2, 4, figsize=(24, 12))
# axes = axes.flatten() # Flatten the subplot axes

# # Feature names
# feature_names = ['Emitter 1', 'Emitter 2', 'Emitter 3', 'Emitter 4', 'Emitter 5', 'Emitter 6', 'Emitter 7', 'Room Number']

# # Plot histogram for each feature
# for i, ax in enumerate(axes):
#     if i < clean_data.shape[1]:
#         ax.hist(clean_data[:, i], bins=20, color='blue', edgecolor='black')
#         ax.set_title(feature_names[i])
#         ax.set_xlabel("Wifi Power")
#         ax.set_ylabel("Number of observations")

# # Remove any extra subplots
# for i in range(clean_data.shape[1], len(axes)):
#     fig.delaxes(axes[i])

# plt.tight_layout()
# plt.show()


# # Analyse the data - noisy data_set

# # Create subplots for histograms
# fig, axes = plt.subplots(2, 4, figsize=(24, 12))
# axes = axes.flatten() # Flatten the subplot axes

# # Feature names
# feature_names = ['Emitter 1', 'Emitter 2', 'Emitter 3', 'Emitter 4', 'Emitter 5', 'Emitter 6', 'Emitter 7', 'Room Number']

# # Plot histogram for each feature
# for i, ax in enumerate(axes):
#     if i < clean_data.shape[1]:
#         ax.hist(noisy_data[:, i], bins=20, color='blue', edgecolor='black')
#         ax.set_title(feature_names[i])
#         ax.set_xlabel("Wifi Power")
#         ax.set_ylabel("Number of observations")

# # Remove any extra subplots
# for i in range(noisy_data.shape[1], len(axes)):
#     fig.delaxes(axes[i])

# plt.tight_layout()
# plt.show()


# In order to have a better understanding of our data, let's compute some statistical information about our data_sets


# Compute statistical information about the clean data_set

print("#######################################################################")
print("\n1. DATA ANALYSIS : Clean data_set\n")

# Calculate statistics
mean_values = np.mean(clean_data, axis=0)
median_values = np.median(clean_data, axis=0)
std_deviation = np.std(clean_data, axis=0)
min_values = np.min(clean_data, axis=0)
max_values = np.max(clean_data, axis=0)

# Print the statistics
print("Mean Values:")
print(mean_values)

print("\nMedian Values:")
print(median_values)

print("\nStandard Deviation:")
print(std_deviation)

print("\nMinimum Values:")
print(min_values)

print("\nMaximum Values:")
print(max_values)

print("\n")
print("#######################################################################")
print("\n")

# Compute statistical information about the noisy data_set

print("#######################################################################")
print("\n1. DATA ANALYSIS : Noisy data_set")
print("\n")

# Calculate statistics
mean_values = np.mean(noisy_data, axis=0)
median_values = np.median(noisy_data, axis=0)
std_deviation = np.std(noisy_data, axis=0)
min_values = np.min(noisy_data, axis=0)
max_values = np.max(noisy_data, axis=0)

# Print the statistics
print("Mean Values:")
print(mean_values)

print("\nMedian Values:")
print(median_values)

print("\nStandard Deviation:")
print(std_deviation)

print("\nMinimum Values:")
print(min_values)

print("\nMaximum Values:")
print(max_values)

print("\n")
print("#######################################################################")


# ## **Step 3 - Creating Decision Trees**

print("\n")
print("Import our methods to create the decision trees...")
print("\n")

# **Functionnal Methods:**
# 
# First of all, let's define functional methods that will help us to build the Tree class and its associated methods


# Functional methods

def p_k(dataset, label):
    """
    Given a column and a label, p_k return the probability of apparition of the label in the column
    
    Args:
    - data_set (matrix)
    - label (int)
    
    Returns:
    - distribution
    """
    
    x_class = dataset[:,-1]==label #extraction fo the last column
    nb_label = np.count_nonzero(x_class)
    nb_total = len(dataset[:,-1])

    if nb_total>0:
        distrib = nb_label/nb_total

    else:
        distrib = 0

    return distrib


def entropy(dataset):
    """
    Given a column and a label k, the method return the entropy of a 
    feature refering to an attribute
    
    Args:
    - data_set (matrix)
    
    Returns:
    - entropy h
    """

    h = 0
    label_unique = list(set(dataset[:, -1]))
    for k in label_unique:
        if p_k(dataset, k)>0:
            h = h + p_k(dataset, k)*np.log2(p_k(dataset, k))

    h = (-1)*h
    return h


def remainder(dataset_right, dataset_left):
    """
    Given two sub-datasets, remainder return the entropy of the remainings datasets
    
    Args:
    - data_right (matrix)
    - data_left (matrix)
    
    Returns:
    - remainder (entropy of the remainings datasets)
    """

    nb_sample_right = len(dataset_right)
    nb_sample_left = len(dataset_left)
    p_left = nb_sample_left/(nb_sample_left + nb_sample_right)
    p_right = (nb_sample_right/(nb_sample_left + nb_sample_right))

    return p_left * entropy(dataset_left) + p_right * entropy(dataset_right)


def gain(dataset_all, dataset_right, dataset_left):
    """
    By definition, this function return the gain value of the dataset for a certain splitting point
    of an attribute
    
    Args:
    - data_set (matrix)
    - data_set_right (matrix)
    - data_set_left (matrix)
    
    Returns:
    - gain
    """

    return entropy(dataset_all) - remainder(dataset_right, dataset_left)


def value_split(dataset, index_attribute):
    """
    Given a dataset and an attribute, the function return the list of all the possible 
    splitting points (discontinuited value in a sorted column)
    
    Args:
    - data_set (matrix)
    - index_attribute (int)
    
    Returns:
    - list_split_points
    """

    attr_x = dataset[:,(index_attribute, -1)]
    list_attr_x = attr_x.tolist()

    list_split_points = []

    list_attr_x = sorted(list_attr_x, key=lambda x: x[0])

    for i in range(0,len(list_attr_x)-1):
        if list_attr_x[i][1] != list_attr_x[i+1][1]:
            avg_value = (list_attr_x[i][0] + list_attr_x[i+1][0]) / 2
            list_split_points.append(avg_value)

    return list_split_points


def find_split(dataset):
    """
    Given a dataset and the number of remainnings attributes, the function returns 
    the index of the attribute selected, the value of the splitting point.
    This method will also return a booleen value, to advertise if a splitting point was found.
    Overall, this function determines the splitting rule for our Decision Tree.
    
    Args:
    - data_set (matrix)
    
    Returns:
    - index_attr (int)
    - value_attr (float)
    - is_found (bool)
    """

    if len(dataset) == 0:
        # Handle the case where the dataset is empty
        return None, None

    nb_attribute = len(dataset[0,:-1])
    rank_gain = 0
    index_attr = 0
    value_attr = 0
    is_found = False
    for i in range(0, nb_attribute):
        split_points = value_split(dataset,i)

        # Check if all values for this attribute are the same
        unique_values = set(dataset[:, i])
        if len(unique_values) == 1:
            continue  # Skip this attribute if all values are the same

        for value in split_points:
            #if value is equal across every rows:
            #if dataset[:,i] 
            
            data_left = dataset[dataset[:,i]<=value]
            data_right = dataset[dataset[:,i]>value]

            if (len(data_left) != 0 and len(data_right) != 0):

                current_gain = gain(dataset, data_left, data_right)
        
                if current_gain >= rank_gain:
                    index_attr = i
                    value_attr = value
                    rank_gain = current_gain
                    is_found = True
                
    return index_attr, value_attr, is_found


# **Node Class:**
# 
# Then, we will define the Node class that returns the instance of a node with various attributes which will help us to define the Tree.


class Node():
    """
    Creates a node for a tree.
    
    Args:
    
    Returns:
    Node
    """
    
    def __init__(self, attribute=None, threshold_split=None, left_node=None, right_node=None, leaf_end=False, depth_input = None, y=None, nb_example_y=0):

        self.node = {
            'attribute': attribute, 
            'value': threshold_split, 
            'left': left_node, 
            'right': right_node, 
            'leaf':leaf_end, 
            'depth':depth_input,
            'label':y,
            'weight_leaf':nb_example_y}

        return    


# **Tree Class:**
# 
# Let's now build our Tree class. We will define a Tree object in order to implement our Decision Tree algorithm to process the data_sets.


class Tree():
    """
    Creates an instance of Tree
    
    Args:

    Returns:
    An instance of Tree
    """

    def __init__(self):
        
        self.root = None
        self.depth_global = 0
        

    # Fitting method
    def decision_tree_learning(self, training_dataset, depth, depth_max=25):
        """
        Recursively constructs a decision tree using the given training dataset and depth.
    
        If all examples in the dataset have the same label, a leaf node is returned.
        Otherwise, the function searches for the best split based on attribute values,
        creates a new decision node, and recursively splits the dataset into two subsets 
        to construct the left and right subtrees.
    
        Args:
        - training_dataset (numpy.ndarray): A 2D array where each row represents a sample and the last column
            contains the label/class of the sample.
        - depth (int): The current depth of the recursion/tree. Starts with 0 for the root.
    
        Returns:
        - Node: The constructed decision or leaf node based on the given dataset.
        - int: The maximum depth of the tree constructed from this dataset.
        """
        
        label_training_dataset = list(set(training_dataset[:,-1]))

        if len(label_training_dataset) == 1: #The subset contains only one label
            return Node(attribute=None, threshold_split=None, left_node=None, right_node=None, leaf_end=True, depth_input = depth, y=label_training_dataset[0], nb_example_y=len(training_dataset)), depth
            
        else:
            #Split condition for a given dataset
            split_attr, split_point, is_found = find_split(training_dataset)

            if is_found == False:
                # Handle the case where no suitable split point can be found for any attribute.
                label_extract = training_dataset[:, -1]
                label_extract = label_extract.astype("int")
                count = np.bincount(label_extract)
                label_extract_large = np.argmax(count)
                return Node(attribute=None, threshold_split=None, left_node=None, right_node=None, leaf_end=True, depth_input=depth, y=label_extract_large, nb_example_y=len(training_dataset)), depth
            
            else:
                #Creation of a node: new decision tree with root as split value
                node_current = Node(split_attr, split_point, Node(), Node(), False, depth)
                
                #Split of the dataset between left and right
                left_dataset = training_dataset[training_dataset[:, split_attr] <= split_point]
                right_dataset = training_dataset[training_dataset[:, split_attr] > split_point]

                
                #Re apply the creation of node for left and right, with a depth+1
                left_branch, left_depth = self.decision_tree_learning(left_dataset, depth+1, depth_max)
                right_branch, right_depth = self.decision_tree_learning(right_dataset, depth+1, depth_max)

                node_current.node["left"] = left_branch
                node_current.node["right"] = right_branch

                self.root = node_current #Root at the end of the recursion
                self.depth_global = max(left_depth, right_depth)
                
                return node_current, max(left_depth, right_depth)


    # Fit method
    def fit_train(self, data):
        """
        Fit a tree with a given data_set vy calling the decision_tree_learning method.

        Args:
        - dataset (numpy.ndarray): A 2D array where each row represents a sample containing features.

        Returns:
        """

        self.root, max_depth = self.decision_tree_learning(data, 1)  


    # Prediction method
    def prediction(self, dataset):
        """
        Predicts the labels for a given dataset based on the decision tree model (after applying decision_tree_learning).

        The function traverses the decision tree starting from the root, following the appropriate branches based on feature values,
        until it reaches a leaf node. The label of the leaf node is then assigned as the prediction for that sample.

        Args:
        - dataset (numpy.ndarray): A 2D array where each row represents a sample containing features.

        Returns:
        - numpy.ndarray: An array containing the predicted labels for each sample in the input dataset.
        """
        
        predictions = []
        for sample in dataset:
            current_node = self.root
            while not current_node.node['leaf']:
                if sample[current_node.node['attribute']] <= current_node.node['value']:
                    current_node = current_node.node['left']
                else:
                    current_node = current_node.node['right']

            predictions.append(current_node.node['label'])

        return np.array(predictions)      


    # Method part of the pruning process: Evaluate the accuracy of the actual Tree.
    def validation_error(self, validation_dataset, accuracy_ref):
        """
        Evaluates the accuracy of the decision tree on a validation dataset 
        and determines if pruning is required.
    
        The function predicts labels for the given validation dataset using the
        actual Tree model. If the computed accuracy is greater than or equal to 
        the provided reference accuracy (`accuracy_ref`), pruning is suggested (returns True), 
        indicating that the node (connected to two leaves) should be cut.
    
        Args:
        - validation_dataset (numpy.ndarray): A 2D array where each row represents 
          a sample with its features and the last column contains the label.
        - accuracy_ref (float): The reference accuracy threshold to decide whether 
          pruning is needed. This is based on a previous evaluation of the original global Tree.
    
        Returns:
        - bool: True if pruning is suggested (current tree's accuracy is greater 
          than or equal to `accuracy_ref`), otherwise False.
        """     

        label_predict = self.prediction(validation_dataset)
        
        # Calculate and store the accuracy or any other performance metric
        accuracy = np.sum(label_predict == validation_dataset[:, -1]) / len(label_predict)
    
        if accuracy >= accuracy_ref:
            return True #True means to cut the leaves
    
        else:
            return False

    # Method part of the pruning process: Cut a node connected to two leafs.
    def tree_cut(self, node_to_cut):
        """
        Prunes the tree by cutting a node connected to two leaf nodes. In other terms,
        the node become a leaf and take the property of the most important leaf.
    
        Args:
        - node_to_cut (Node): The node object (part of the object Tree) connected to two leaf nodes that is to be pruned.
    
        Returns:
        - tuple: 
          - int: The original attribute number of `node_to_cut` before the cut.
          - float: The original value associated to the attribute of `node_to_cut` before the cut.
    
        Note:
        save_attribute and save_value are returned in case we need to reverse the cut and recreate this node (eg. if validation_error return False).
        """

        node_to_cut.node["leaf"] = True

        save_attribute = node_to_cut.node["attribute"]
        save_value = node_to_cut.node["value"]
    
        label_weight_left = node_to_cut.node["left"].node["weight_leaf"]
        label_weight_right = node_to_cut.node["right"].node["weight_leaf"]
    
        if label_weight_left > label_weight_right:
            node_to_cut.node["attribute"] = node_to_cut.node["left"].node["attribute"]
            node_to_cut.node["value"] = node_to_cut.node["left"].node["value"]
    
        else:
            node_to_cut.node["attribute"] = node_to_cut.node["right"].node["attribute"]
            node_to_cut.node["value"] = node_to_cut.node["right"].node["value"]
    
        return save_attribute, save_value

    
    # Method part of the pruning process: Reverse the process of Node cut.
    def reverse_tree_cut(self, node_to_reverse, save_1, save_2):
        """
        Reverses the pruning process for a previously cut node.
    
        This method undoes the changes made by the `tree_cut` method, essentially 
        restoring the node to its state before the pruning.
    
        Args:
        - node_to_reverse (Node): The node object that was previously pruned and needs to be restored.
        - save_1 (int): The original attribute number of `node_to_reverse` that needs to be restored.
        - save_2 (float): The original value associated to the attribute of `node_to_reverse` that needs to be restored.
    
        Returns:
        None
        """
        node_to_reverse.node["leaf"] = False

        node_to_reverse.node["attribute"] = save_1
        node_to_reverse.node["value"] = save_2
        
        return
    

    # Method part of the pruning process: Compute the pruning with a recursive search.
    def search_path_leaf(self, node_start, validation_dataset, accuracy_ref):
        """
        Recursively searches the decision tree for pruning candidates starting from a given node.
    
        The method explores the tree to identify nodes that are connected to two leaf nodes.
        Once a candidate is identified, the method prunes it using the `tree_cut` method and validates the modified tree using 
        the `validation_error` method. If the pruned tree does not improve (or is worse) based 
        on the provided accuracy reference, the pruning is reversed using the `reverse_tree_cut` method.
    
        Args:
        - node_start (Node): The starting node for the search. This node is the root of the Tree
        - validation_dataset (numpy.ndarray): A 2D array where each row represents a sample with 
          its features and the last column contains the label.
        - accuracy_ref (float): The reference accuracy threshold against which the performance 
          of the pruned tree is compared.
    
        Returns:
        None
        """
        
        if node_start is None or node_start.node["leaf"] is True: #case where the start node is None (beginning issue) or if we search on a leaf (after a node connected to node and leaf)
                return node_start
    
        if node_start.node["left"].node["leaf"] == True and node_start.node["right"].node["leaf"]: #condition d'arret recursif => implies a test for transformation
            # Should cut the tree
            prop1, prop2 = self.tree_cut(node_start)
            if self.validation_error(validation_dataset, accuracy_ref) == False: #if improvement is TRUE
                self.reverse_tree_cut(node_start, prop1, prop2)
         
        else:
            if node_start.node['left'] is not None:
                left_result = self.search_path_leaf(node_start.node['left'], validation_dataset, accuracy_ref)
    
            if node_start.node['right'] is not None:
                right_result = self.search_path_leaf(node_start.node['right'], validation_dataset, accuracy_ref)
    
        return


#Additional exploration of the dataset => verify if two rows are equal (same attributes values) but with different label
def check_duplicate_rows_with_different_labels(dataset):
    # Get the rows and labels
    rows = dataset[:, :-1]  # Exclude the last column (labels)
    labels = dataset[:, -1]

    num_rows, num_columns = rows.shape

    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            if np.array_equal(rows[i], rows[j]) and labels[i] != labels[j]:
                return True

    return False


# ## **Step 4 - Evaluation**


# Now that our algorithm is set up, let's define our methods and metrics in order to evaluate our predictions.


# **Cross Validation**:
# 
# First, we will define our two methods in order to process a cross validation, namely: the simple Cross-validation and the Nested Cross-validation


def get_tree_depth(node):
    if node is None:
        return 0
    left_depth = get_tree_depth(node.node['left'])
    right_depth = get_tree_depth(node.node['right'])
    return max(left_depth, right_depth) + 1


def cross_validate_without_pruning(data, k_folds, random_seed=None):
    """
    Perform a single k-fold cross-validation without tree pruning.

    Args:
        data (numpy.ndarray): A 2D array where each row represents a sample containing features.
        k_folds (int): The number of folds for cross-validation.

    Returns:
        Tuple:
        - y_gold_matrix (List): A list to store the gold labels for each fold.
        - y_prediction_matrix (List): A list to store prediction labels for each fold.
        - tree_depths (List): A list to store the tree depth for each fold.
    """
        
    # Set the random seed if provided
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random

    # Shuffle the data
    rng.shuffle(data)
    
    y_gold_matrix = []  # List to store gold labels for each folds
    y_prediction_matrix = []  # List to store predictions labels for each folds
    tree_depths = []  # List to store tree depth for each fold

    # Split data into k folds
    fold_partitions = np.array_split(data, k_folds)
    for fold_index in range(len(fold_partitions)):
        # Take out the final test partition and put it to the side
        final_test = fold_partitions[fold_index]
        # Remove the final_test partition of the dataset
        training_partitions = np.concatenate([part for index, part in enumerate(fold_partitions) if index != fold_index])

        # Build a new model for each fold
        my_tree_exp = Tree()  # Reset the model
        my_tree_exp.fit_train(training_partitions)  # Train the model
        y_gold_matrix.append(final_test[:, -1])  # Stock the gold labels for this fold
        predictions = my_tree_exp.prediction(final_test)  # Make predictions
        y_prediction_matrix.append(predictions)  # Stock the prediction labels for this fold

        # Compute and store the tree depth
        tree_depth = get_tree_depth(my_tree_exp.root)
        tree_depths.append(tree_depth)

    return y_gold_matrix, y_prediction_matrix, tree_depths




def cross_validation_with_pruning(data, k_folds, random_seed=None):
    """
    Perform nested k-fold cross-validation with tree pruning.

    Args:
        data (numpy.ndarray): A 2D array where each row represents a sample containing features.
        k_folds (int): The number of folds for cross-validation.

    Returns:
        Tuple:
        - y_gold_matrix (List): A list of lists to store the gold labels for each fold.
        - y_prediction_matrix (List): A list of lists to store prediction labels for each fold.
        - pruning_rounds (List): A list to store the number of pruning rounds for each fold.
        - depth_pruning (List): A list to store the depth of the tree after pruning for each fold.
    """
    
    # Set the random seed if provided
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random

    # Shuffle the data
    rng.shuffle(data)
    
    # Initialize lists to store gold labels and predictions
    y_gold_matrix = []
    y_prediction_matrix = []

    y_gold_fold = []
    y_predictions_fold = []

    pruning_rounds = []
    depth_pruning = []

    fold_partitions = np.array_split(data, k_folds)

    for test_fold_index in range(k_folds):
        test_fold = fold_partitions[test_fold_index]
        train_prune_folds = [fold_partitions[i] for i in range(k_folds) if i != test_fold_index]

        for validation_fold_index in range(k_folds - 1):
            validation_fold = train_prune_folds[validation_fold_index]

            # Train and prune the tree with k-2 training folds
            my_tree_exp = Tree()
            training_data = np.concatenate([fold for i, fold in enumerate(train_prune_folds) if i != validation_fold_index])
            
            my_tree_exp.fit_train(training_data)

            # Initialisation of the pruning
            prediction_ref = my_tree_exp.prediction(validation_fold)
            gold_ref = validation_fold[:, -1]
            accuracy_ref = np.sum(prediction_ref == gold_ref) / len(gold_ref)
            
            is_done = False
            n_cut = 0

            while not is_done:
                # Perform pruning based on validation fold
                my_tree_exp.search_path_leaf(my_tree_exp.root, validation_fold, accuracy_ref)
                prediction_prune = my_tree_exp.prediction(validation_fold)
                gold_validation = validation_fold[:, -1]
                accuracy_new_prune = np.sum(prediction_prune == gold_ref) / len(gold_validation)

                n_cut += 1

                if (accuracy_new_prune == accuracy_ref):
                    pruning_rounds.append(n_cut)
                    depth_pruning.append(my_tree_exp.depth_global)
                    is_done = True

            # Predict the labels for the test fold
            test_data = test_fold[:, :-1]  # Exclude the last column (labels)
            predictions = my_tree_exp.prediction(test_data)

            # Store the gold labels and predictions
            y_gold_fold.append(test_fold[:, -1])
            y_predictions_fold.append(predictions)

        y_gold_matrix.append(y_gold_fold)
        y_prediction_matrix.append(y_predictions_fold)

        y_gold_fold = []
        y_predictions_fold = []

    return y_gold_matrix, y_prediction_matrix, pruning_rounds, depth_pruning


# **Metrics methods**:
# 
# First, we will define our metrics in order to evaluate a tree, namely: the confusion matrix, the accuracy, the recall/precision rates and the F1-measure

print("Import our metrics to evaluate the decision trees...")
print("\n")

# Metrics methods

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix given a y_label vector and a y_prediction vector

    Args: y_true, y_pred

    Returns: confusion_matrix
    """

    labels = np.unique(y_true)
    # Create the confusion matrix
    cm = np.zeros((len(labels), len(labels),), dtype=int)

    # Fill the confusion matrix
    for i in range(len(y_true)):
        true_label = np.where(labels == y_true[i])
        pred_label = np.where(labels == y_pred[i])
        cm[true_label, pred_label] += 1

    return cm


def accuracy(confusion_matrix):
    """
    Compute the accuracy of a prediction given the confusion_matrix

    Args: confusion_matrix

    Returns: confusion_matrix
    """

    n = len(confusion_matrix)
    acc=0
    tot=0
    for i in range (n):
        for j in range (n):
            if (i==j):
                acc=acc+confusion_matrix[i][j]
            tot=tot+confusion_matrix[i][j]
    return(acc/tot)

def class_accuracy(confusion_matrix):
    """
    Compute the accuracy for each class given the confusion_matrix

    Args:
        confusion_matrix: A 2D array representing the confusion matrix.

    Returns:
        class_accuracies: A list of accuracies for each class.
    """

    n = len(confusion_matrix)
    class_accuracies = []

    for i in range(n):
        correct = confusion_matrix[i][i]
        total = sum(confusion_matrix[i])
        accuracy = correct / total if total > 0 else 0.0
        class_accuracies.append(accuracy)

    return class_accuracies


def calculate_precision_recall(confusion_matrix):
    """
    Compute the accuracy of a prediction given the confusion_matrix

    Args: 
    - confusion_matrix

    Returns: 
    - Precision
    - Recall
    """

    num_classes = len(confusion_matrix)
    precision = [0] * num_classes
    recall = [0] * num_classes

    for class_index in range(num_classes):
        true_positives = confusion_matrix[class_index][class_index]
        false_positives = sum(confusion_matrix[i][class_index] for i in range(num_classes) if i != class_index)
        false_negatives = sum(confusion_matrix[class_index][i] for i in range(num_classes) if i != class_index)

        if true_positives + false_positives > 0:
            precision[class_index] = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives > 0:
            recall[class_index] = true_positives / (true_positives + false_negatives)

    return (precision, recall)

def calculate_f1_measure(precision, recall):
    f1_scores = []
    for i in range(len(precision)):
        if precision[i] + recall[i] == 0:
            f1_scores.append(0)  # Avoid division by zero
        else:
            f1_scores.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
    
    return f1_scores


# As required in the assignemnt, here is bellow a method named **evaluate** that computes the accuracy of a given tree.


def evaluate(test_db, trained_tree):
    """
    Evaluate the accuracy of a trained decision tree on a test dataset.

    Args:
        test_db (numpy.ndarray): A 2D array representing the test dataset, where each row represents a sample containing features.
        trained_tree: A trained decision tree to make predictions.

    Returns:
        float: The accuracy of the trained decision tree on the test dataset.
    """

    y_predictions = trained_tree.prediction(test_db)
    y_gold = test_db[:, -1]
    accuracy = np.sum(y_predictions == y_gold) / len(y_gold)
    return accuracy


# We also added functions to be alble to save the data as csv files for our report. We have 2 without pruning and 2 with pruning to get the global confusion matrix and the measures in a Matrix

# def save_combined_confusion_matrices_to_csv(matrix_list, i):
#     # Sum all the confusion matrices
#     combined_matrix = sum(matrix_list)
    
#     # Normalize the combined matrix
#     normalized_matrix = combined_matrix / combined_matrix.sum(axis=1, keepdims=True)

#     # Create a list to store data for the CSV file
#     header = ["Class"]
#     for j in range(1, len(combined_matrix[0]) + 1):
#         header.append("Class " + str(j))
#     csv_data = [header]

#     # Round the values in the normalized matrix to four decimal places and add to the list
#     for j, row in enumerate(normalized_matrix):
#         class_name = "Class " + str(j + 1)
#         row_data = [class_name] + [f"{value:.4f}" for value in row]  # Format values with four decimal places
#         csv_data.append(row_data)

#     # Save the data to a CSV file
#     file_name = f"combined_matrix_confusion_{i}.csv"
#     with open(file_name, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(csv_data)


# def create_and_save_metrics_matrix(precisions, recalls, f1_measures, accuracies, class_accuracies, i):
#     # Calculate the mean and standard deviation of metrics for each class
#     average_precision = np.mean(precisions, axis=0)
#     average_recall = np.mean(recalls, axis=0)
#     average_f1 = np.mean(f1_measures, axis=0)
#     average_class_accuracy = np.mean(class_accuracies, axis=0)

#     std_dev_precision = np.std(precisions, axis=0)
#     std_dev_recall = np.std(recalls, axis=0)
#     std_dev_f1 = np.std(f1_measures, axis=0)
#     std_dev_class_accuracy = np.std(class_accuracies, axis=0)

#     # Round all values to 4 decimal places
#     average_precision = np.round(average_precision, 4)
#     std_dev_precision = np.round(std_dev_precision, 4)
#     average_recall = np.round(average_recall, 4)
#     std_dev_recall = np.round(std_dev_recall, 4)
#     average_f1 = np.round(average_f1, 4)
#     std_dev_f1 = np.round(std_dev_f1, 4)
#     average_class_accuracy = np.round(average_class_accuracy, 4)
#     std_dev_class_accuracy = np.round(std_dev_class_accuracy, 4)

#     # Create the metrics matrix
#     matrix_measure = [
#         ["Metric"] + [f"Class {j}" for j in range(1, len(average_precision) + 1)],
#         ["Average Precision"] + list(average_precision),
#         ["Std Dev Precision"] + list(std_dev_precision),
#         ["Average Recall"] + list(average_recall),
#         ["Std Dev Recall"] + list(std_dev_recall),
#         ["Average F1-Measure"] + list(average_f1),
#         ["Std Dev F1"] + list(std_dev_f1),
#         ["Average Class Accuracy"] + list(average_class_accuracy),
#         ["Std Dev Class Accuracy"] + list(std_dev_class_accuracy)
#     ]

#     # Save the data to a CSV file
#     file_name = f"metrics_matrix_{i}.csv"
#     with open(file_name, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(matrix_measure)
        



# ## **Step 5 - Execution**


# ### **A - Without pruning**


# **Clean data_set:** Without pruning

print("#######################################################################")
print("DECISION TREE RESULTS\n")
print("WITHOUT PRUNING\n")
print("CLEAN DATA_SET\n")

# Format precision for displaying results
precision_format = "{:.4f}"

# Define the number of folds for cross-validation
k_folds = 10

# Set a random seed for reproducibility (change to None if you do not want to set a seed)
Seed = 70050

# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
depth_scores = []
class_accuracies_scores = []

l_matrix_confusion = []

# Perform k-fold cross-validation without pruning
y_gold_matrix, y_prediction_matrix, depth_values = cross_validate_without_pruning(clean_data, k_folds, Seed)

print("Results for the clean data_set without pruning:\n")

# Loop through each fold for evaluation
for i in range(k_folds):

    # Compute confusion matrix
    cm = confusion_matrix(y_gold_matrix[i], y_prediction_matrix[i])
    l_matrix_confusion.append(cm)

    # Compute accuracy
    acc = accuracy(cm)

    # Compute precision and recall
    precision, recall = calculate_precision_recall(cm)

    # Compute F1 score
    f1 = calculate_f1_measure(precision, recall)

    # Compute Class accuracy
    class_accuracies = class_accuracy(cm)

    # Store the metrics for this fold
    accuracy_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    class_accuracies_scores.append(class_accuracies)
    
    # Append depth value
    depth_scores.append(depth_values[i])

    # Display results for this fold
    print(f"Fold {i + 1} Results (Clean Data_set / Without Pruning):")
    print(f"Confusion Matrix: \n {cm}")
    print("Accuracy: {:.4f}".format(acc))
    print("Accuracy for Each Class:", " ".join(["{:.4f}".format(acc) for acc in class_accuracies]))
    print("Precision:", end=" ")
    for p in precision:
        print(f"{precision_format.format(p)}", end=" ")
    print()
    print("Recall:", end=" ")
    for r in recall:
        print(f"{precision_format.format(r)}", end=" ")
    print()
    print("F1 Score:", end=" ")
    for f in f1:
        print(f"{f:.4f}", end=" ")
    print()
    print(f"Depth: {depth_values[i]:.4f}\n")

# Calculate and display the mean and standard deviation of metrics for all folds
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

mean_precision = np.mean(precision_scores, axis=0)
std_precision = np.std(precision_scores, axis=0)

mean_recall = np.mean(recall_scores, axis=0)
std_recall = np.std(recall_scores, axis=0)

mean_f1 = np.mean(f1_scores, axis=0)
std_f1 = np.std(f1_scores, axis=0)

mean_class_accuracy = np.mean(class_accuracies_scores, axis=0)
std_class_accuracy = np.std(class_accuracies_scores, axis=0)

# Calculate the mean and standard deviation of depth
mean_depth = np.mean(depth_scores)
std_depth = np.std(depth_scores)

# Format the lists as strings without brackets
mean_class_accuracy_str = ", ".join(["{:.4f}".format(acc) for acc in mean_class_accuracy])
std_class_accuracy_str = ", ".join(["{:.4f}".format(acc) for acc in std_class_accuracy])

print("Overall Results (On the clean dataset, without Pruning):\n")
print(f"Mean Accuracy: {mean_accuracy:.4f} (Std: {std_accuracy:.4f})")
print(f"Mean Accuracy for Each Class: {mean_class_accuracy_str}")
print(f"Std Accuracy for Each Class: {std_class_accuracy_str}")
print("Mean Precision:", end=" ")
for p in mean_precision:
    print(f"{p:.4f}", end=" ")
print()
print("Std Precision:", end=" ")
for p in std_precision:
    print(f"{p:.4f}", end=" ")
print()
print("Mean Recall:", end=" ")
for r in mean_recall:
    print(f"{r:.4f}", end=" ")
print()
print("Std Recall:", end=" ")
for r in std_recall:
    print(f"{r:.4f}", end=" ")
print()
print("Mean F1 Score:", end=" ")
for f in mean_f1:
    print(f"{f:.4f}", end=" ")
print()
print("Std F1 Score:", end=" ")
for f in std_f1:
    print(f"{f:.4f}", end=" ")
print()
print(f"Mean Depth: {mean_depth:.4f} (Std: {std_depth:.4f})")

# save_combined_confusion_matrices_to_csv(l_matrix_confusion, 1)
# create_and_save_metrics_matrix(precision_scores, recall_scores, f1_scores, accuracy_scores, class_accuracies_scores, 1)

print("#######################################################################")
print("\nEND OF PROCESSING\n")

# **Noisy data_set:** Without pruning

print("#######################################################################")
print("DECISION TREE RESULTS\n")
print("WITHOUT PRUNING\n")
print("NOISY DATA_SET\n")

# Format precision for displaying results
precision_format = "{:.4f}"

# Define the number of folds for cross-validation
k_folds = 10

# Set a random seed for reproducibility (change to None if you do not want to set a seed)
Seed = 70050

# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
depth_scores = []
class_accuracies_scores = []

l_matrix_confusion = []

# Perform k-fold cross-validation without pruning
y_gold_matrix, y_prediction_matrix, depth_values = cross_validate_without_pruning(noisy_data, k_folds, Seed)

print("Results for the noisy data_set without pruning:\n")

# Loop through each fold for evaluation
for i in range(k_folds):

    # Compute confusion matrix
    cm = confusion_matrix(y_gold_matrix[i], y_prediction_matrix[i])
    l_matrix_confusion.append(cm)

    # Compute accuracy
    acc = accuracy(cm)

    # Compute precision and recall
    precision, recall = calculate_precision_recall(cm)

    # Compute F1 score
    f1 = calculate_f1_measure(precision, recall)

    # Compute Class accuracy
    class_accuracies = class_accuracy(cm)

    # Store the metrics for this fold
    accuracy_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    class_accuracies_scores.append(class_accuracies)
    
    # Append depth value
    depth_scores.append(depth_values[i])

    # Display results for this fold
    print(f"Fold {i + 1} Results (Noisy data_set / Without Pruning):")
    print(f"Confusion Matrix: \n {cm}")
    print("Accuracy: {:.4f}".format(acc))
    print("Accuracy for Each Class:", " ".join(["{:.4f}".format(acc) for acc in class_accuracies]))
    print("Precision:", end=" ")
    for p in precision:
        print(f"{precision_format.format(p)}", end=" ")
    print()
    print("Recall:", end=" ")
    for r in recall:
        print(f"{precision_format.format(r)}", end=" ")
    print()
    print("F1 Score:", end=" ")
    for f in f1:
        print(f"{f:.4f}", end=" ")
    print()
    print(f"Depth: {depth_values[i]:.4f}\n")

# Calculate and display the mean and standard deviation of metrics for all folds
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

mean_precision = np.mean(precision_scores, axis=0)
std_precision = np.std(precision_scores, axis=0)

mean_recall = np.mean(recall_scores, axis=0)
std_recall = np.std(recall_scores, axis=0)

mean_f1 = np.mean(f1_scores, axis=0)
std_f1 = np.std(f1_scores, axis=0)

mean_class_accuracy = np.mean(class_accuracies_scores, axis=0)
std_class_accuracy = np.std(class_accuracies_scores, axis=0)

# Calculate the mean and standard deviation of depth
mean_depth = np.mean(depth_scores)
std_depth = np.std(depth_scores)

# Format the lists as strings without brackets
mean_class_accuracy_str = ", ".join(["{:.4f}".format(acc) for acc in mean_class_accuracy])
std_class_accuracy_str = ", ".join(["{:.4f}".format(acc) for acc in std_class_accuracy])

print("Overall Results (Noisy data_set / Without Pruning):\n")
print(f"Mean Accuracy: {mean_accuracy:.4f} (Std: {std_accuracy:.4f})")
print(f"Mean Accuracy for Each Class: {mean_class_accuracy_str}")
print(f"Std Accuracy for Each Class: {std_class_accuracy_str}")
print("Mean Precision:", end=" ")
for p in mean_precision:
    print(f"{p:.4f}", end=" ")
print()
print("Std Precision:", end=" ")
for p in std_precision:
    print(f"{p:.4f}", end=" ")
print()
print("Mean Recall:", end=" ")
for r in mean_recall:
    print(f"{r:.4f}", end=" ")
print()
print("Std Recall:", end=" ")
for r in std_recall:
    print(f"{r:.4f}", end=" ")
print()
print("Mean F1 Score:", end=" ")
for f in mean_f1:
    print(f"{f:.4f}", end=" ")
print()
print("Std F1 Score:", end=" ")
for f in std_f1:
    print(f"{f:.4f}", end=" ")
print()
print(f"Mean Depth: {mean_depth:.4f} (Std: {std_depth:.4f})")

# save_combined_confusion_matrices_to_csv(l_matrix_confusion, 2)
# create_and_save_metrics_matrix(precision_scores, recall_scores, f1_scores, accuracy_scores, class_accuracies_scores, 2)

print("#######################################################################")
print("\nEND OF PROCESSING\n")

# ### **B - With pruning**


# **Clean data_set:** With pruning

print("#######################################################################")
print("DECISION TREE RESULTS\n")
print("WITH PRUNING\n")
print("CLEAN DATA_SET\n")

# Format precision for displaying results
precision_format = "{:.4f}"

# Define the number of folds for cross-validation
k_folds = 10

# Set a random seed for reproducibility (change to None if you don't want to set a seed)
Seed = 70050

# Initialize lists to store metrics
accuracy_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []
class_accuracies_per_fold = []
confusion_matrices = []
depth_pruning_per_fold = []

# Perform nested k-fold cross-validation with pruning
y_gold_matrix_pruning, y_prediction_matrix_pruning, pruning_rounds_fold, depth_pruning_fold = cross_validation_with_pruning(clean_data, k_folds, Seed)

print("Results for the clean data_set with pruning:\n")

# Loop through each fold for evaluation
for i in range(k_folds):
    fold_accuracy = 0
    fold_precision = [0] * np.unique(clean_data[:, -1])
    fold_recall = [0] * np.unique(clean_data[:, -1])
    fold_f1 = [0] * np.unique(clean_data[:, -1])
    fold_class_accuracies = [0] * np.unique(clean_data[:, -1])
    fold_confusion_matrix = np.zeros((4, 4), dtype=int)
    fold_depth_pruning = []

    # Loop through each validation fold within the nested cross-validation
    for j in range(k_folds - 1):
        y_true = y_gold_matrix_pruning[i][j]
        y_pred = y_prediction_matrix_pruning[i][j]

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fold_confusion_matrix += cm

        # Compute accuracy
        fold_accuracy += accuracy(cm)

        # Compute precision and recall
        precision, recall = calculate_precision_recall(cm)
        fold_precision = [p + q for p, q in zip(fold_precision, precision)]
        fold_recall = [r + s for r, s in zip(fold_recall, recall)]

        # Compute F1 score
        f1 = calculate_f1_measure(precision, recall)
        fold_f1 = [f + g for f, g in zip(fold_f1, f1)]

        # Compute class accuracies
        class_accuracies = class_accuracy(cm)
        fold_class_accuracies = [c + d for c, d in zip(fold_class_accuracies, class_accuracies)]
        
        # Append depth pruning values
        fold_depth_pruning.append(depth_pruning_fold[i * (k_folds - 1) + j])

    # Average the metrics for the current fold
    fold_accuracy /= (k_folds - 1)
    fold_precision = [p / (k_folds - 1) for p in fold_precision]
    fold_recall = [r / (k_folds - 1) for r in fold_recall]
    fold_f1 = [f / (k_folds - 1) for f in fold_f1]
    fold_class_accuracies = [c / (k_folds - 1) for c in fold_class_accuracies]

    accuracy_per_fold.append(fold_accuracy)
    precision_per_fold.append(fold_precision)
    recall_per_fold.append(fold_recall)
    f1_per_fold.append(fold_f1)
    class_accuracies_per_fold.append(fold_class_accuracies)
    confusion_matrices.append(fold_confusion_matrix)
    depth_pruning_per_fold.append(fold_depth_pruning)

    # Display results for this fold
    print(f"Fold {i + 1} Results (Clean data_set / With Pruning):")
    print(f"Confusion Matrix:\n{fold_confusion_matrix}")
    print(f"Accuracy: {fold_accuracy:.4f}")
    print("Accuracy for Each Class:", " ".join(["{:.4f}".format(acc) for acc in fold_class_accuracies]))
    print("Precision:", end=" ")
    for p in fold_precision:
        print(f"{precision_format.format(p)}", end=" ")
    print()
    print("Recall:", end=" ")
    for r in fold_recall:
        print(f"{precision_format.format(r)}", end=" ")
    print()
    print("F1 Score:", end=" ")
    for f in fold_f1:
        print(f"{f:.4f}", end=" ")
    print()
    print(f"Depth Pruning: {np.mean(fold_depth_pruning):.4f}\n")
    
# Calculate and display the mean and standard deviation of metrics for all folds

mean_accuracy = np.mean(accuracy_per_fold)
std_accuracy = np.std(accuracy_per_fold)

mean_precision = np.mean(precision_per_fold, axis=0) / k_folds
std_precision = np.std(precision_per_fold, axis=0) / k_folds

mean_recall = np.mean(recall_per_fold, axis=0) / k_folds
std_recall = np.std(recall_per_fold, axis=0) / k_folds

mean_f1 = np.mean(f1_per_fold, axis=0) / k_folds
std_f1 = np.std(f1_per_fold, axis=0) / k_folds

mean_class_accuracies = np.mean(class_accuracies_per_fold, axis=0) / k_folds
mean_depth_pruning = np.mean(depth_pruning_per_fold)

print("Overall Results (Clean data_set / With Pruning):")
print(f"Mean Accuracy: {mean_accuracy:.4f} (Std: {std_accuracy:.4f})")
print(f"Mean Accuracy for Each Class: {', '.join([f'{acc:.4f}' for acc in mean_class_accuracies])}")
print("Mean Precision:", end=" ")
for p in mean_precision:
    print(f"{p:.4f}", end=" ")
print()
print("Std Precision:", end=" ")
for p in std_precision:
    print(f"{p:.4f}", end=" ")
print()
print("Mean Recall:", end=" ")
for r in mean_recall:
    print(f"{r:.4f}", end=" ")
print()
print("Std Recall:", end=" ")
for r in std_recall:
    print(f"{r:.4f}", end=" ")
print()
print("Mean F1 Score:", end=" ")
for f in mean_f1:
    print(f"{f:.4f}", end=" ")
print()
print(f"Mean Depth Pruning: {mean_depth_pruning:.4f}")


# save_combined_confusion_matrices_to_csv(confusion_matrices, 3)
# create_and_save_metrics_matrix(precision_per_fold, recall_per_fold, f1_per_fold, accuracy_per_fold, class_accuracies_per_fold, 3)

print("#######################################################################")
print("\nEND OF PROCESSING\n")

# **Noisy data_set:** With pruning

print("#######################################################################")
print("DECISION TREE RESULTS\n")
print("WITH PRUNING\n")
print("NOISY DATA_SET\n")

# Define the number of folds for cross-validation
k_folds = 10

# Set a random seed for reproducibility (change to None if you don't want to set a seed)
Seed = 70050

# Initialize lists to store metrics
accuracy_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []
class_accuracies_per_fold = []
confusion_matrices = []
depth_pruning_per_fold = []

# Perform nested k-fold cross-validation with pruning
y_gold_matrix_pruning, y_prediction_matrix_pruning, pruning_rounds_fold, depth_pruning_fold = cross_validation_with_pruning(noisy_data, k_folds, Seed)

print("Results for the noisy data_set with pruning:\n")

# Loop through each fold for evaluation
for i in range(k_folds):
    fold_accuracy = 0
    fold_precision = [0] * np.unique(noisy_data[:, -1])
    fold_recall = [0] * np.unique(noisy_data[:, -1])
    fold_f1 = [0] * np.unique(noisy_data[:, -1])
    fold_class_accuracies = [0] * np.unique(noisy_data[:, -1])
    fold_confusion_matrix = np.zeros((4, 4), dtype=int)
    fold_depth_pruning = []

    # Loop through each validation fold within the nested cross-validation
    for j in range(k_folds - 1):
        y_true = y_gold_matrix_pruning[i][j]
        y_pred = y_prediction_matrix_pruning[i][j]

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fold_confusion_matrix += cm

        # Compute accuracy
        fold_accuracy += accuracy(cm)

        # Compute precision and recall
        precision, recall = calculate_precision_recall(cm)
        fold_precision = [p + q for p, q in zip(fold_precision, precision)]
        fold_recall = [r + s for r, s in zip(fold_recall, recall)]

        # Compute F1 score
        f1 = calculate_f1_measure(precision, recall)
        fold_f1 = [f + g for f, g in zip(fold_f1, f1)]

        # Compute class accuracies
        class_accuracies = class_accuracy(cm)
        fold_class_accuracies = [c + d for c, d in zip(fold_class_accuracies, class_accuracies)]

        # Append depth pruning values
        fold_depth_pruning.append(depth_pruning_fold[i * (k_folds - 1) + j])

    # Average the metrics for the current fold
    fold_accuracy /= (k_folds - 1)
    fold_precision = [p / (k_folds - 1) for p in fold_precision]
    fold_recall = [r / (k_folds - 1) for r in fold_recall]
    fold_f1 = [f / (k_folds - 1) for f in fold_f1]
    fold_class_accuracies = [c / (k_folds - 1) for c in fold_class_accuracies]

    accuracy_per_fold.append(fold_accuracy)
    precision_per_fold.append(fold_precision)
    recall_per_fold.append(fold_recall)
    f1_per_fold.append(fold_f1)
    class_accuracies_per_fold.append(fold_class_accuracies)
    confusion_matrices.append(fold_confusion_matrix)
    depth_pruning_per_fold.append(fold_depth_pruning)

    # Display results for this fold
    print(f"Fold {i + 1} Results (Noisy data_set / With Pruning):")
    print(f"Confusion Matrix:\n{fold_confusion_matrix}")
    print(f"Accuracy: {fold_accuracy:.4f}")
    print("Accuracy for Each Class:", " ".join(["{:.4f}".format(acc) for acc in fold_class_accuracies]))
    print("Precision:", end=" ")
    for p in fold_precision:
        print(f"{precision_format.format(p)}", end=" ")
    print()
    print("Recall:", end=" ")
    for r in fold_recall:
        print(f"{precision_format.format(r)}", end=" ")
    print()
    print("F1 Score:", end=" ")
    for f in fold_f1:
        print(f"{f:.4f}", end=" ")
    print()
    print(f"Depth Pruning: {np.mean(fold_depth_pruning):.4f}\n")

# Calculate and display the mean and standard deviation of metrics for all folds

mean_accuracy = np.mean(accuracy_per_fold)
std_accuracy = np.std(accuracy_per_fold)

mean_precision = np.mean(precision_per_fold, axis=0) / k_folds
std_precision = np.std(precision_per_fold, axis=0) / k_folds

mean_recall = np.mean(recall_per_fold, axis=0) / k_folds
std_recall = np.std(recall_per_fold, axis=0) / k_folds

mean_f1 = np.mean(f1_per_fold, axis=0) / k_folds
std_f1 = np.std(f1_per_fold, axis=0) / k_folds

mean_class_accuracies = np.mean(class_accuracies_per_fold, axis=0) / k_folds
mean_depth_pruning = np.mean(depth_pruning_per_fold)

print("Overall Results (Noisy data_set / With Pruning):")
print(f"Mean Accuracy: {mean_accuracy:.4f} (Std: {std_accuracy:.4f})")
print(f"Mean Accuracy for Each Class: {', '.join([f'{acc:.4f}' for acc in mean_class_accuracies])}")
print("Mean Precision:", end=" ")
for p in mean_precision:
    print(f"{p:.4f}", end=" ")
print()
print("Std Precision:", end=" ")
for p in std_precision:
    print(f"{p:.4f}", end=" ")
print()
print("Mean Recall:", end=" ")
for r in mean_recall:
    print(f"{r:.4f}", end=" ")
print()
print("Std Recall:", end=" ")
for r in std_recall:
    print(f"{r:.4f}", end=" ")
print()
print("Mean F1 Score:", end=" ")
for f in mean_f1:
    print(f"{f:.4f}", end=" ")
print()
print(f"Mean Depth Pruning: {mean_depth_pruning:.4f}")

# save_combined_confusion_matrices_to_csv(confusion_matrices, 4)
# create_and_save_metrics_matrix(precision_per_fold, recall_per_fold, f1_per_fold, accuracy_per_fold, class_accuracies_per_fold, 4)

print("#######################################################################")
print("\nEND OF PROCESSING\n")

# ## **Bonus - Tree Visualisation**

print("#######################################################################")
print("BONUS\n")
print("TREE VISUALISATION\n")
print("#######################################################################")
print("\n")

# Plot the entire tree trained on the clean_data
level_coords = {}

def plot_tree(node, x, y, dx, dy, parent_x=None, parent_y=None):
    if node.node['leaf']:
        box = plt.Rectangle((x-0.01, y-0.01), 0.02, 0.02, fill=True, facecolor='red', edgecolor='red')
        plt.gca().add_patch(box)
        plt.text(x, y, f'Label: {node.node["label"]}', fontsize=8, ha='center', va='center', color='black')

    else:
        # Draw a box for non-leaf nodes
        box = plt.Rectangle((x-0.01, y-0.01), 0.02, 0.02, fill=True, facecolor='lightblue', edgecolor='blue')
        plt.gca().add_patch(box)
        # Print the attribute and threshold inside the box
        attribute = node.node['attribute']
        threshold = node.node['value']
        plt.text(x, y, f'Attribute: {attribute},\n x < {threshold}', fontsize=8, ha='center', va='center', color='black')

        # Store the coordinates of this level
        if y not in level_coords:
            level_coords[y] = []
        level_coords[y].append((x-0.01, y-0.01))

        if parent_x is not None and parent_y is not None:
            # Draw an arrow from the parent to the current node
            px, py = parent_x, parent_y
            ax.annotate('', xy=(x, y+0.01), xytext=(px, py-0.01), arrowprops={'arrowstyle': '->', 'color': 'blue'})

        # Calculate coordinates for the left and right child nodes
        x_left_leaf = x - dx/3
        x_right_leaf = x + dx/3
        y_next_leaf = y - dy/2

        x_left = x - dx
        x_right = x + dx
        y_next = y - dy

        if node.node['left'].node['leaf'] and node.node['right'].node['leaf']:
            plot_tree(node.node['left'], x_left_leaf, y_next_leaf, dx, dy, x, y)
            plot_tree(node.node['right'], x_right_leaf, y_next_leaf, dx, dy, x, y)

        elif node.node['right'].node['leaf']:
            plot_tree(node.node['left'], x_left, y_next, dx, dy, x, y)
            plot_tree(node.node['right'], x_right_leaf, y_next_leaf, dx, dy, x, y)

        elif node.node['left'].node['leaf']:
            plot_tree(node.node['left'], x_left_leaf, y_next_leaf, dx, dy, x, y)
            plot_tree(node.node['right'], x_right, y_next, dx, dy, x, y)

        else:
            plot_tree(node.node['left'], x_left, y_next, dx, dy, x, y)
            plot_tree(node.node['right'], x_right, y_next, dx, dy, x, y)

fig, ax = plt.subplots(figsize=(50, 50))
ax.set_axis_off()

# Initialize your tree and data
#tree = Tree()
#tree.fit_train(clean_data)

# Start plotting the tree
#plot_tree(tree.root, 0.7, 0.96, 0.07, 0.07)

#plt.savefig("decision_tree.png", dpi=300, bbox_inches='tight')

#plt.show()

