#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python
from IDAPICourseworkLibrary import *
from numpy import *
import copy
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
    for j in range(len(theData)):
        temp = theData[j][root]
        prior[temp] = prior[temp] + 1

    #for i in range(noStates[root]):
        #prior[i] = prior[i]/len(theData)
    prior = prior/len(theData)
# Coursework 1 task 1 should be inserted here

# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
    numbVarP = zeros((noStates[varP]), float)
# Coursework 1 task 2 should be inserte4d here
    numbVarP = zeros((noStates[varP]), float)
    for j in range(len(theData)):
        temp1 = theData[j][varC]
        temp2 = theData[j][varP]
        numbVarP[temp2] = numbVarP[temp2]+1
        cPT[temp1][temp2] = cPT[temp1][temp2] + 1

    for k in range(noStates[varP]):
        for i in range(noStates[varC]):
            if numbVarP[k] != 0:
                cPT[i][k] = cPT[i][k]/numbVarP[k]
    # end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here
    for j in range(len(theData)):
        temp1 = theData[j][varRow]
        temp2 = theData[j][varCol]
        jPT[temp1][temp2] = jPT[temp1][temp2] + (1/float(len(theData)))
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here
    sum_cols_aJPT = zeros(len(aJPT[0]), float)
# coursework 1 taks 4 ends here
    for j in range(len(aJPT[0])):
        for i in range(len(aJPT)):
            sum_cols_aJPT[j] = sum_cols_aJPT[j] + aJPT[i][j]

        for k in range(len(aJPT)):
            aJPT[k][j] = aJPT[k][j]/sum_cols_aJPT[j]
    return aJPT
#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    #rootPdf = zeros(4, float)
    rootPdf = copy.deepcopy(naiveBayes[0])
    #print "NB:"
    #print naiveBayes[0]
# Coursework 1 task 5 should be inserted here
    for i in range(len(rootPdf)):
        #print rootPdf[i]
        for j in range(len(theQuery)):
            #print (naiveBayes[1+j][theQuery[j]][i])
            rootPdf[i] = rootPdf[i]*(naiveBayes[1+j][theQuery[j]][i])
    #print "rootpdf"
    #print rootPdf
    alpha = sum(rootPdf)
    #print alpha
    rootPdf = rootPdf/alpha

# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
    rows, cols = jP.shape
    row_sums = zeros(rows, float)
    col_sums = zeros(cols, float)
    for i in range(rows):
        for j in range(cols):
            row_sums[i] = row_sums[i] + jP[i][j]
            col_sums[j] = col_sums[j] + jP[i][j]


    for i in range(rows):
        for j in range(cols):
            temp = jP[i][j]/(row_sums[i]*col_sums[j])
            if temp == 0 or row_sums[i] == 0 or col_sums[j]==0:
                temp = 1

            mi = mi + jP[i][j]*log2(temp)


# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for i in range(noVariables):
        for j in range(i+1, noVariables):
            MIMatrix[i][j] = MutualInformation(JPT(theData, i, j, noStates))
            MIMatrix[j][i] = MIMatrix[i][j]
# end of coursework 2 task 2
    return MIMatrix


# Function to compute an ordered list of dependencies
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    rows,cols = depMatrix.shape
    for i in range(rows):
        for j in range(i+1, noVariables):
            obj = [depMatrix[i][j], i, j]
            depList.append(obj)
    print depList
    depList = sorted(depList, reverse = True)
    print depList
# end of coursework 2 task 3
    return array(depList)
#


# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    rows,cols = depList.shape
    spanningTree = []
    connections = zeros((noVariables, noVariables), float)
    for i in range(rows):
        var1 = depList[i][1]
        var2 = depList[i][2]
        if connections[var1][var2]==0:
            spanningTree.append(depList[i])
            connections[var1][var2] = 1
            connections[var2][var1] = 1
            for k in range(noVariables):
                temp =[]
                for l in range(noVariables):
                    if connections[k][l] == 1:
                        temp.append(l)
                for n in range(len(temp)):
                    for m in range(n+1, len(temp)):
                        connections[temp[n]][temp[m]] = 1
                        connections[temp[m]][temp[n]] = 1

    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
    prob_parent = zeros([noStates[parent1], noStates[parent2]],float)
    rows,cols = theData.shape

    for j in range(rows):
        cPT[theData[j][child]][theData[j][parent1]][theData[j][parent2]] = cPT[theData[j][child]][theData[j][parent1]][theData[j][parent2]] + 1
        prob_parent[theData[j][parent1]][theData[j][parent2]] =  prob_parent[theData[j][parent1]][theData[j][parent2]] + 1

    for k in range(noStates[child]):
        for i in range(noStates[parent1]):
            for l in range(noStates[parent2]):
                if prob_parent[i][l] != 0:
                    cPT[k][i][l] = cPT[k][i][l]/prob_parent[i][l]


# End of Coursework 3 task 1
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here
def HepC_Network(theData, noStates):
    arcList= [[0], [1], [2, 0], [3, 4], [4, 1], [5, 4], [6, 1], [7, 0, 1], [8, 7]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT(theData, 3, 4, noStates)
    cpt4 = CPT(theData, 4, 1, noStates)
    cpt5 = CPT(theData, 5, 4, noStates)
    cpt6 = CPT(theData, 6, 1, noStates)
    cpt7 = CPT_2(theData, 7, 0, 1, noStates)
    cpt8 = CPT(theData, 8, 7, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, cpt7, cpt8]
    return arcList, cptList

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
##def MDLSize(arcList, cptList, noDataPoints, noStates):
##    mdlSize = 0.0
### Coursework 3 task 3 begins here
##    for j in cptList:
##        dim = j.shape
##        if len(dim) == 1:
##            mdlSize =  mdlSize + (dim[0]-1)
##        elif len(dim) == 2:
##            mdlSize =  mdlSize + (dim[0]*dim[1]-dim[0])
##        elif len(dim) == 3:
##            mdlSize =  mdlSize + (dim[0]*dim[1]*dim[2]-dim[0])
##        else:
##            print("More than two parents detected")
##
##        s = mdlSize*(log2(noDataPoints))/2
### Coursework 3 task 3 ends here
##    return s

def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here
    for j in cptList:
       dims  = j.shape
       par_sum = dims[0] - 1
       for k in dims[1:]:
             par_sum= par_sum* k
       mdlSize = mdlSize + par_sum

    mdlSize = mdlSize*(log2(noDataPoints)/2)
    return mdlSize
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
    for j in range(len(cptList)):
        #print jP
        #print j
        if len(arcList[j]) == 1:
            #print("Entered 1 clause")
            #print len(arcList[j])
            #print cptList[j][dataPoint[arcList[j][0]]]
            jP = jP*cptList[j][dataPoint[arcList[j][0]]]
        elif len(arcList[j]) == 2:
            #print("Entered 2 clause")
            #print len(arcList[j])
            #print cptList[j][dataPoint[arcList[j][0]]][dataPoint[arcList[j][1]]]
            jP = jP*cptList[j][dataPoint[arcList[j][0]]][dataPoint[arcList[j][1]]]
        elif len(arcList[j]) == 3:
            #print("Entered 3 clause")
            #print len(arcList[j])
            #print cptList[j][dataPoint[arcList[j][0]]][dataPoint[arcList[j][1]]][dataPoint[arcList[j][2]]]
            jP = jP*cptList[j][dataPoint[arcList[j][0]]][dataPoint[arcList[j][1]]][dataPoint[arcList[j][2]]]
        else:
            print("Node has to many ancestors")
# Coursework 3 task 4 ends here
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
    #prod_joint_p = 1
    for j in theData:
        mdlAccuracy = mdlAccuracy + log2(JointProbability(j, arcList, cptList))
    #mdlAccuracy = log2(prod_joint_p)

# Coursework 3 task 5 ends here
    return mdlAccuracy


# Function to calculate the best network for a data set by deleting a node at a time
def best_scoring_network(theData, arcList, cptList, noStates, noDataPoints):
    first = 1
    for j in arcList:
        cptList_temp = copy.deepcopy(cptList)
        arcList_temp = copy.deepcopy(arcList)
        cptList_temp2 = copy.deepcopy(cptList)
        arcList_temp2 = copy.deepcopy(arcList)
        if len(j) == 2:
            cptList_temp[j[0]] = Prior(theData, j[0], noStates)
            arcList_temp[j[0]] = [j[0]]
            #mdl_acc_temp = MDLAccuracy(theData, arcList, cptList)
            score_temp = MDLSize(arcList_temp, cptList_temp, noDataPoints, noStates) - MDLAccuracy(theData, arcList_temp, cptList_temp)
            if first == 1:
                #mdl_acc = mdl_acc_temp
                best_score = score_temp
                cpt_best = cptList_temp
                arc_best = arcList_temp
                first = 0
            else:
                if score_temp < best_score:
                    #mdl_acc = mdl_acc_temp
                    best_score = score_temp
                    cpt_best = cptList_temp
                    arc_best = arcList_temp

        elif len(j) == 3:
            cptList_temp[j[0]] = CPT(theData, j[0], j[1], noStates)
            cptList_temp.append(Prior(theData, j[0], noStates))
            arcList_temp[j[0]] = [j[0], j[1]]
            arcList_temp.append([j[0]])
            #mdl_acc_temp = MDLAccuracy(theData, arcList, cptList)
            score_temp = MDLSize(arcList_temp, cptList_temp, noDataPoints, noStates) - MDLAccuracy(theData, arcList_temp, cptList_temp)

            cptList_temp2[j[0]] = CPT(theData, j[0], j[2], noStates)
            cptList_temp2.append(Prior(theData, j[0], noStates))
            arcList_temp2[j[0]] = [j[0], j[2]]
            arcList_temp2.append([j[2]])
            #mdl_acc_temp2 = MDLAccuracy(theData, arcList, cptList)
            score_temp2 = MDLSize(arcList_temp2, cptList_temp2, noDataPoints, noStates) - MDLAccuracy(theData, arcList_temp2, cptList_temp2)

            if first == 1:
                #mdl_acc = mdl_acc_temp
                best_score = score_temp
                cpt_best = cptList_temp
                arc_best = arcList_temp
                first = 0
                if score_temp2 < best_score:
                    #mdl_acc = mdl_acc_temp2
                    best_score = score_temp2
                    cpt_best = cptList_temp2
                    arc_best = arcList_temp2
            else:
                if score_temp < best_score:
                    #mdl_acc = mdl_acc_temp
                    best_score = score_temp
                    cpt_best = cptList_temp
                    arc_best = arcList_temp

                if score_temp2 < best_score:
                    #mdl_acc = mdl_acc_temp2
                    best_score = score_temp2
                    cpt_best = cptList_temp2
                    arc_best = arcList_temp2

        else:
            print("Node does not have either 1 or 2 parents")


    return cpt_best, arc_best, best_score

# End of coursework 3
#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending
    # order of their eignevalues magnitudes


    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#
##noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
##theData = array(datain)

##AppendString("results.txt","Coursework One Results by Michael Murray")
##AppendString("results.txt","") #blank line
##
##AppendString("results.txt","The prior probability of node 0")
##prior = Prior(theData, 0, noStates)
###print prior
##AppendList("results.txt", prior)
##
##AppendString("results.txt","The conditional probability table between nodes 0 (the root) and 2")
##lmat2= CPT(theData, 2, 0, noStates)
###print lmat2
##AppendArray("results.txt", lmat2)
##
##AppendString("results.txt","The joint probability table between nodes 0 (the root) and 2")
##jpt2 = JPT(theData, 2, 0, noStates)
###print jpt2
##AppendArray("results.txt", jpt2)
##
##AppendString("results.txt","The normalized joint probability table between nodes 0 (the root) and 2")
##norm_jpt2 = JPT2CPT(jpt2)
###print norm_jpt2
##AppendArray("results.txt", norm_jpt2)
##
##
###Calculate link tables
##lmat1 = CPT(theData, 1, 0, noStates)
##lmat3 = CPT(theData, 3, 0, noStates)
##lmat4 = CPT(theData, 4, 0, noStates)
##lmat5 = CPT(theData, 5, 0, noStates)
##naiveBayes = [prior, lmat1, lmat2, lmat3, lmat4, lmat5]
###define instantiations
##ins1 = [4,0,0,0,5]
##ins2 = [6,5,2,5,5]
##
##print naiveBayes[0]
##AppendString("results.txt","The probability distribution over the root node given query [4,0,0,0,5] ")
##root_prob1 = Query(ins1, naiveBayes)
##print root_prob1
##AppendList("results.txt", root_prob1)
##AppendString("results.txt","The probability distribution over the root node given query [6,5,2,5,5] ")
##print naiveBayes[0]
##root_prob2 = Query(ins2, naiveBayes)
##print root_prob2
##print 'new line'
##AppendList("results.txt", root_prob2)
##
###
### continue as described
###
###

###Coursework 2
##noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
##theData = array(datain)
##
##AppendString("results2.txt","Coursework Two Results by Michael Murray")
##AppendString("results2.txt","") #blank line
##
##AppendString("results2.txt","Dependency matrix for the HepatitisC data set:")
##HC_Dep = DependencyMatrix(theData, noVariables, noStates)
##AppendArray("results2.txt", HC_Dep)
##print HC_Dep
##
##AppendString("results2.txt","") #blank line
##AppendString("results2.txt","Dependency list for the HepatitisC data set:")
##HC_Dep_List = DependencyList(HC_Dep)
##AppendArray("results2.txt", HC_Dep_List)
##print HC_Dep_List
##
##
##AppendString("results2.txt","") #blank line
##AppendString("results2.txt","The spanning tree found for the HepatitisC data seta set:")
##span_tree_HC = SpanningTreeAlgorithm(HC_Dep_List, noVariables)
##AppendArray("results2.txt", span_tree_HC)
##print("Spanning Tree:")
##print span_tree_HC

#Coursework 3:
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
#AppendString("results3.txt","Coursework Two Results by Michael Murray")
#AppendString("results3.txt","") #blank line
arcList, cptList = HepC_Network(theData, noStates)

print cptList
