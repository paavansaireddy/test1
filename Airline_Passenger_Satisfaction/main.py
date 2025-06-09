#!/usr/bin/env python
# coding: utf-8

# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;"><h1>Airline passenger satisfaction</h1></div>

# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;"><h1>Reading data</h1></div>

# In[1]:


import numpy as np
import pandas as pd
import graphviz
import collections
import math
import seaborn as sns
import sklearn
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve,roc_auc_score, auc, precision_score, f1_score, recall_score, RocCurveDisplay
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# read data file
data = pd.read_csv("https://drive.google.com/uc?export=download&id=1JF5JMnp4kWV8Fe4ltWjMOloF8x8PSIbq")


# Table dimensions:

# In[3]:


#number of columns and rows
data.shape


# In[4]:


#sample data
data.head()


# Now, take a close look at the dataset:

# In[5]:


data.info()


# In[6]:


data = data.drop(data.iloc[:,[0, 1]], axis = 1)


# In[7]:


categoricalFeatures = [0, 1, 3, 4] + list(range(6, 20))
data.iloc[:,categoricalFeatures] = data.iloc[:,categoricalFeatures].astype('category')


# Now lets take a look back into the dataset:

# In[8]:


data.info()


# The last column satisfaction is the output variable

# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;"><h1>Data visualization</h1></div>

# In[9]:


data.describe()


# In[10]:


data.describe(include = ['category'])


# In[11]:


pt=data.satisfaction.value_counts()
plt.pie(pt, autopct = '%1.1f%%',labels = ["Neutral or dissatisfied", "Satisfied"])
pass


# From the above pie chart we can infer that the <b>the selection is almost balanced</b>.
# 
# correlation matrix for the features is:

# In[12]:


correl_matrix = data.corr(numeric_only=True)
correl_matrix


# In[13]:


sns.heatmap(correl_matrix, square = True)
pass


# In[14]:


correl_matrix.where(np.triu(correl_matrix > 0.5, k=1)).stack().sort_values(ascending = False)


# In[15]:


plt.scatter(data['Arrival Delay in Minutes'], data['Departure Delay in Minutes'], alpha = 0.5)
pass


# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;"><h1>Filling in missing values</h1></div>

# In[16]:


data.isna().sum()


# In[17]:


data['Arrival Delay in Minutes'].fillna(data['Arrival Delay in Minutes'].median(axis = 0), inplace = True)


# In[18]:


data.isna().sum()


# In[19]:


data.dtypes


# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;">
#     <h1>Handling categorical features</h1>
# </div>

# In[20]:


df = pd.DataFrame(data)
numerical = df.select_dtypes(include=['number']).columns
df[numerical] = df[numerical].astype(float)


# In[21]:


data.dtypes


# In[22]:


numerical = [c for c in data.columns if data[c].dtype.name != 'category']
numerical.remove('satisfaction')
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'category']


# In[23]:


data.dtypes


# In[24]:


binary_columns = ['Gender', 'Customer Type', 'Type of Travel']
# binary_columns = [c for c in binary_columns if c not in ['Class', 'satisfaction']]
nonbinary_columns = ['Class', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']

print(binary_columns, nonbinary_columns)


# In[25]:


for col in binary_columns:
    print(col, ': ', end = '')
    for uniq in data[col].unique():
        if uniq == data[col].unique()[-1]:
            print(uniq, end = '.')
        else:
            print(uniq, end = ', ')
    print()


# In[26]:


data[col] == uniq


# In[27]:


# Define the mappings for each binary column
mappings = {
    'Gender': {'Male': 0, 'Female': 1},
    'Customer Type': {'Loyal Customer': 0, 'disloyal Customer': 1},
    'Type of Travel': {'Personal Travel': 0, 'Business travel': 1}
}

# Apply the mappings to the binary columns
for col in binary_columns:
    if col in mappings:
        data[col] = data[col].map(mappings[col])
        print(f"Column: {col}")
        print(f"Data type: {type(data[col])}")
        print(data[col].describe(), end='\n\n')
        for uniq in data[col].unique():
            print(uniq)


# In[28]:


data[nonbinary_columns]


# In[29]:


data_nonbinary = pd.get_dummies(data[nonbinary_columns])
print(data_nonbinary.columns)


# In[30]:


len(data_nonbinary.columns)


# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;">
#     <h1>Normalization of quantitative features</h1>
# </div>

# In[31]:


col_num = data[numerical]
col_num


# In[32]:


col_num = col_num.drop('Gender', axis=1)
col_num = col_num.drop('Customer Type', axis=1)
col_num = col_num.drop('Type of Travel', axis=1)
col_num = col_num.drop('Class', axis=1)


# In[33]:


col_num


# In[34]:


# Standardize each numerical column in the DataFrame
for col in col_num.columns:
    col_num[col] = (col_num[col] - col_num[col].mean()) / col_num[col].std()



# In[35]:


col_num.describe()


# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;">
#     <h1>Table formation</h1>
# </div>

# In[36]:


target = data['satisfaction']
data = pd.concat((col_num, data_nonbinary, data[binary_columns]), axis = 1)


# In[37]:


data.describe()


# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;">
#     <h1>Data Splitting</h1>
# </div>

# In[38]:


y = target
X = data


# In[39]:


X.columns


# In[40]:


y


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 777)


# In[42]:


bagging_train = []
bagging_test = []


# In[43]:


# Assuming X_train and X_test are pandas DataFrames
X_train = X_train.astype(float)
X_test = X_test.astype(float)


# In[44]:


# Define a mapping for the string values to numeric
mapping = {'neutral or dissatisfied': 0.0, 'satisfied': 1.0}

# Apply the mapping to y_train
y_train = y_train.replace(mapping).astype(float)


# In[45]:


y_train


# In[46]:


# Define a mapping for the string values to numeric
mapping = {'neutral or dissatisfied': 0.0, 'satisfied': 1.0}

# Apply the mapping to y_train
y_test = y_test.replace(mapping).astype(float)


# In[47]:


X_train


# In[48]:


# Loop through the columns and print each data type
for col in X_test.columns:
    print(f"{col}: {X_test[col].dtype}")


# In[49]:


print(y_train.dtype)


# In[50]:


# Convert DataFrames to NumPy arrays
xtrain = X_train.values
xtest = X_test.values
ytrain = y_train.values
ytest = y_test.values


# In[51]:


xtest, _, ytest, _ = train_test_split(xtest, ytest, test_size=8/9, random_state=42)
xtrain, X_rest, ytrain, y_rest = train_test_split(xtrain, ytrain, test_size=8/9, random_state=42)


# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;">
#     <h1>Logistic Regression Model</h1>
# </div>
# 

# In[52]:


class LogRegModel:

    def __init__(self, rate=0.01, iterations=1000):
        self.rate = rate
        self.iterations = iterations
        self.coef_ = None
        self.intercept_ = None

    def fit_model(self, features, labels):
        num_samples, num_features = features.shape
        self.coef_ = np.zeros(num_features)
        self.intercept_ = 0

        for _ in range(self.iterations):
            model_output = np.dot(features, self.coef_) + self.intercept_
            predicted_labels = self._logistic_function(model_output)

            gradient_w = (1 / num_samples) * np.dot(features.T, (predicted_labels - labels))
            gradient_b = (1 / num_samples) * np.sum(predicted_labels - labels)
            self.coef_ -= self.rate * gradient_w
            self.intercept_ -= self.rate * gradient_b

    def predict_model(self, features):
        model_output = np.dot(features, self.coef_) + self.intercept_
        predicted_labels = self._logistic_function(model_output)
        predicted_class = [1 if i > 0.5 else 0 for i in predicted_labels]
        return np.array(predicted_class)

    def _logistic_function(self, z):
        return 1 / (1 + np.exp(-z))

def model_accuracy(actual, predicted):
    correct_predictions = np.sum(actual == predicted) / len(actual)
    return correct_predictions

# Example usage:
iterations_record = []
accuracy_record = []
log_reg_model = LogRegModel(rate=0.0001, iterations=1000)
log_reg_model.fit_model(xtrain, ytrain)
model_predictions = log_reg_model.predict_model(xtest)
iterations_record.append(1000)
print("Logistic Regression Analysis:")
print("LogReg Model classification accuracy:", model_accuracy(ytest, model_predictions))
accuracy_record.append(model_accuracy(ytest, model_predictions))


# In[53]:


confusion_mat = confusion_matrix(ytest, model_predictions)
print("Conf Matrix:\n", confusion_mat)


class_rep = classification_report(ytest, model_predictions)
print("Detailed Classification Report:\n", class_rep)

# Compute the Receiver Operating Characteristic (ROC) curve data points
false_pos_rate, true_pos_rate, threshold_values = roc_curve(ytest, model_predictions)

# Calculate the Area Under the Curve (AUC) for the ROC
area_under_curve = auc(false_pos_rate, true_pos_rate)

# Plot the ROC curve using matplotlib
plt.figure(figsize=(8, 6))
plt.plot(false_pos_rate, true_pos_rate, color='purple', lw=2, label='ROC Curve (AUC = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1], color='lightgrey', lw=2, linestyle='--')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('(FPR)')
plt.ylabel('(TPR)')
plt.title('ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;">
#     <h1>Decision Tree</h1>
# </div>
# 

# In[54]:


import numpy as np
import random

def partitionDataset(df, splitRatio):
    if isinstance(splitRatio, float):
        splitRatio = round(splitRatio * len(df))
    allIndices = df.index.tolist()
    testIdx = random.sample(population=allIndices, k=splitRatio)
    testSet = df.loc[testIdx]
    trainingSet = df.drop(testIdx)
    return trainingSet, testSet

def isHomogeneous(samples):
    if len(np.unique(samples[:, -1])) == 1:
        return True
    else:
        return False

def determineClass(samples):
    classes, countsOfClasses = np.unique(samples[:, -1], return_counts=True)
    return classes[countsOfClasses.argmax()]

def findSplits(samples, randomFeats):
    splits = {}
    _, totalColumns = samples.shape
    columnIdx = list(range(totalColumns - 1))
    if randomFeats is not None and len(randomFeats) <= len(columnIdx):
        columnIdx = randomFeats
    for col in columnIdx:
        colValues = samples[:, col]
        uniqueVals = np.unique(colValues)
        if len(uniqueVals) == 1:
            splits[col] = uniqueVals
        else:
            splits[col] = []
            for index in range(len(uniqueVals)):
                if index != 0:
                    currentVal = uniqueVals[index]
                    previousVal = uniqueVals[index - 1]
                    splits[col].append((currentVal + previousVal) / 2)
    return splits

def divideData(samples, divisionCol, divisionVal):
    colValues = samples[:, divisionCol]
    return samples[colValues <= divisionVal], samples[colValues > divisionVal]

def computeEntropy(samples):
    _, counts = np.unique(samples[:, -1], return_counts=True)
    probs = counts / counts.sum()
    return sum(probs * -np.log2(probs))

def totalEntropy(lowerData, upperData):
    probLower = len(lowerData) / (len(lowerData) + len(upperData))
    probUpper = len(upperData) / (len(lowerData) + len(upperData))
    return probLower * computeEntropy(lowerData) + probUpper * computeEntropy(upperData)
import numpy as np
import random

def identifyOptimalDivision(dataset, possibleDivisions, numRandomDivisions=None):
    minEntropy = float('inf')
    optimalDivisionAttr = None
    optimalDivisionValue = None
    if numRandomDivisions is None:
        for divisionAttr in possibleDivisions:
            for divisionValue in possibleDivisions[divisionAttr]:
                lowerSubset, upperSubset = divideData(dataset, divisionAttr, divisionValue)
                entropy = totalEntropy(lowerSubset, upperSubset)
                if entropy < minEntropy:
                    minEntropy = entropy
                    optimalDivisionAttr = divisionAttr
                    optimalDivisionValue = divisionValue
    else:
        for _ in range(numRandomDivisions):
            attr = random.choice(list(possibleDivisions))
            value = random.choice(possibleDivisions[attr])
            lowerSubset, upperSubset = divideData(dataset, attr, value)
            entropy = totalEntropy(lowerSubset, upperSubset)
            if entropy < minEntropy:
                minEntropy = entropy
                optimalDivisionAttr = attr
                optimalDivisionValue = value
    return optimalDivisionAttr, optimalDivisionValue

def constructTree(df, depth=0, minSize=2, maxDepth=1000, selectedAttributes=None, numRandomDivisions=None):
    if depth == maxDepth or len(df) < minSize:
        return determineClass(df)
    if depth == 0:
        global ATTR_NAMES
        ATTR_NAMES = df.columns
        dataset = df.values
        if selectedAttributes is not None and selectedAttributes <= len(ATTR_NAMES) - 1:
            selectedAttributes = random.sample(population=list(range(len(ATTR_NAMES) - 1)), k=selectedAttributes)
        else:
            selectedAttributes = None
    else:
        dataset = df
    if isHomogeneous(dataset) or len(dataset) < minSize or depth == maxDepth:
        return determineClass(dataset)
    else:
        depth += 1
        possibleDivisions = findSplits(dataset, selectedAttributes)
        divisionAttr, divisionValue = identifyOptimalDivision(dataset, possibleDivisions, numRandomDivisions)
        lowerSubset, upperSubset = divideData(dataset, divisionAttr, divisionValue)
        if len(lowerSubset) == 0 or len(upperSubset) == 0:
            return determineClass(dataset)
        else:
            query = f"{ATTR_NAMES[divisionAttr]} <= {divisionValue}"
            subtree = {query: []}
            positiveBranch = constructTree(lowerSubset, depth, minSize, maxDepth, selectedAttributes, numRandomDivisions)
            negativeBranch = constructTree(upperSubset, depth, minSize, maxDepth, selectedAttributes, numRandomDivisions)
            if positiveBranch == negativeBranch:
                subtree = positiveBranch
            else:
                subtree[query].append(positiveBranch)
                subtree[query].append(negativeBranch)
            return subtree

def evaluateSample(instance, tree):
    if not isinstance(tree, dict):
        return tree
    query = list(tree.keys())[0]
    attr, threshold = query.split(" <= ")
    if instance[attr] <= float(threshold):
        response = tree[query][0]
    else:
        response = tree[query][1]
    return evaluateSample(instance, response)

def treePredictions(df, tree):
    outcomes = df.apply(evaluateSample, axis=1, args=(tree,))
    return outcomes

def computeAccuracy(predictions, trueLabels):
    correctPredictions = predictions == trueLabels
    return correctPredictions.mean()


# In[ ]:


print("Decision Tree Analysis:")

depth_level = 4

# Construct the tree model
tree_model = constructTree(X_train, maxDepth=depth_level)


test_results = treePredictions(X_test, tree_model)
test_accuracy = computeAccuracy(test_results, y_test) * 100

# Output the test accuracy
print(f"Depth Level = {depth_level}: ", end="")
print(f"Test Accuracy = {test_accuracy:.2f}%")


# In[ ]:


con_mat = confusion_matrix(y_test, test_results)
print("Confusion Matrix:\n", con_mat)


class_rep = classification_report(y_test, test_results)
print("Detailed Classification Report:\n", class_rep)


false_positive_rate, true_positive_rate, threshold_values = roc_curve(y_test, test_results)


area_under_curve = auc(false_positive_rate, true_positive_rate)


plt.figure(figsize=(8, 6))
plt.plot(false_positive_rate, true_positive_rate, color='purple', lw=2, label='ROC (AUC = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1], color='lightgrey', lw=2, linestyle='--')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('(FPR)')
plt.ylabel('(TPR)')
plt.title('ROC for Decision Tree')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;">
#     <h1>Support Vector Machine</h1>
# </div>

# In[ ]:


class SupportVectorMachine:
    def __init__(self, rate=0.001, regularization=0.01, iterations=1000):
        self.rate = rate
        self.regularization = regularization
        self.iterations = iterations
        self.weights = None
        self.intercept = None

    def train(self, features, targets):
        num_samples, num_features = features.shape
        targets_ = np.where(targets <= 0, -1, 1)
        self.weights = np.zeros(num_features)
        self.intercept = 0

        for _ in range(self.iterations):
            for index, feature_vector in enumerate(features):
                condition = targets_[index] * (np.dot(feature_vector, self.weights) - self.intercept) >= 1
                if condition:
                    self.weights -= self.rate * (2 * self.regularization * self.weights)
                else:
                    self.weights -= self.rate * (2 * self.regularization * self.weights - feature_vector * targets_[index])
                    self.intercept -= self.rate * targets_[index]

    def compute_decision(self, features):
        return np.dot(features, self.weights) - self.intercept

    def classify(self, features):
        decision = np.dot(features, self.weights) - self.intercept
        return np.sign(decision)

    def evaluate_accuracy(self, features, targets):
        predictions = self.classify(features)
        return np.mean(predictions == targets)


svm_classifier = SupportVectorMachine()
feature_train, feature_rest, target_train, target_rest = train_test_split(xtrain, ytrain, test_size=8/9, random_state=42)
svm_classifier.train(feature_train, target_train)


decision_scores = svm_classifier.compute_decision(xtest)


false_positive_rate, true_positive_rate, threshold_vals = roc_curve(ytest, decision_scores)
print("Support Vector Machine Analysis:")

plt.figure(figsize=(8, 8))
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance Level')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM Classifier')
plt.legend(loc='lower right')
plt.show()


area_under_curve = roc_auc_score(ytest, decision_scores)
print("Area Under Curve (AUC):", area_under_curve)


# In[67]:


predicted_targets = svm_classifier.classify(xtest)

classifier_accuracy = svm_classifier.evaluate_accuracy(xtest, ytest)
print("Classifier Accuracy:", classifier_accuracy)


confusion_mat = confusion_matrix(ytest, predicted_targets)
print("Confusion Matrix:\n", confusion_mat)


classification_rep = classification_report(ytest, predicted_targets)
print("Classification Report:")
print(classification_rep)


# <div style="border: 3px solid #556b2f; border-radius: 20px; background: linear-gradient(to right, #fdf5e6, #faebd7); text-align: center; font-family: 'Times New Roman', Times, serif; color: #556b2f; padding: 15px; margin: 10px;">
#     <h1>K-Nearest Neighbors</h1>
# </div>

# In[68]:


class NearestNeighbors:
    """
    Nearest Neighbors algorithm for classification and regression tasks.
    """

    def __init__(self, neighbors=5, classify=True):
        """
        Initialize the Nearest Neighbors model with the number of neighbors and task type.

        Parameters:
            neighbors: Number of nearest neighbors to use for predictions.
            classify: Boolean indicating if the task is classification (True) or regression (False).
        """
        self.neighbors = neighbors
        self.classify = classify
        self.features = None
        self.labels = None

    def train(self, features_train, labels_train):
        """
        Memorize the training dataset.

        Parameters:
            features_train: Training dataset features (2D array).
            labels_train: Training dataset labels (1D array).
        """
        self.features = features_train
        self.labels = labels_train

    def calculate_distance(self, point1, point2):
       
        distance = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
        return math.sqrt(distance)

    def classify_point(self, point):
        """
        Classify or predict the value of a new data point using the trained neighbors.

        Parameters:
            point: Data point to classify or predict (vector).

        Returns:
            The predicted class or value for the data point.
        """
        distances = [(self.calculate_distance(point, feature), index)
                     for index, feature in enumerate(self.features)]
        
        # Sort by distance and select the nearest neighbors
        distances.sort()
        neighbors_indices = [index for _, index in distances[:self.neighbors]]

        # Predict the majority class for classification tasks
        if self.classify:
            majority_vote = collections.Counter(self.labels[index] for index in neighbors_indices)
            return majority_vote.most_common(1)[0][0]

        # Predict the average value for regression tasks
        else:
            return sum(self.labels[index] for index in neighbors_indices) / self.neighbors


# In[69]:


# Initialize and train the Nearest Neighbors model
nn_model = NearestNeighbors(neighbors=5, classify=True)
nn_model.train(xtrain, ytrain)


predicted_labels = [nn_model.classify_point(feature) for feature in xtest]
print("KNN Analysis:")

model_accuracy = accuracy_score(ytest, predicted_labels)
print("Model Accuracy:", model_accuracy)


conf_mat = confusion_matrix(ytest, predicted_labels)
print("Confusion Matrix:\n", conf_mat)




# In[70]:


if len(np.unique(ytest)) == 2:  
    
    fpr, tpr, thresholds = roc_curve(ytest, predicted_targets)
 
    roc_auc_value = auc(fpr, tpr)


    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC Curve (AUC = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance Level')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve Analysis')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# # Create a detailed classification report
# detailed_classification_report = classification_report(ytest, predicted_targets, digits=4)
# print("Detailed Classification Report:")
# print(detailed_classification_report)

