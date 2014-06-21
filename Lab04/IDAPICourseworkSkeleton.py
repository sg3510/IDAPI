#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python
from IDAPICourseworkLibrary import *
from numpy import *
import copy
import gc
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    #theData = theData.astype(int)
    prior = bincount(theData[0:,root])
    prior = prior.astype(float)
    #prior = divide(prior,noStates)
    prior = divide(prior,sum(prior))
# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
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
##    i = 0
##    for x in unique(theData[:,varP]):
##        index = argwhere(theData[0:,varP] == x)
##        val = asarray(theData[index,varC].flatten().tolist())
##        #print val
##        cptcol = bincount(val,minlength=noStates[varC]).astype(float)
##        cptcol = divide(cptcol,sum(cptcol))
##        #print cptcol.T
##        cPT[:,i] = cptcol
##        i=i+1
##        #print bincount(theData[index,varC])
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here
    i = 0
    for x in unique(theData[:,varCol]):
        index = argwhere(theData[0:,varCol] == x)
        val = asarray(theData[index,varRow].flatten().tolist())
        #print val
        jptcol = bincount(val,minlength=noStates[varRow]).astype(float)
        #jptcol = divide(jptcol,sum(jptcol))
        #print jptcol.T
        jPT[:,i] = jptcol
        i=i+1
    jPT = divide(jPT,sum(jPT))
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here

    rows, cols = aJPT.shape
    for col in xrange(cols):
        aJPT[:,col] /= sum(aJPT[:,col])
# coursework 1 taks 4 ends here
    return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    #print naiveBayes[0]
    #print len(naiveBayes)
    query_ans = 0;
    for x in range(len(naiveBayes[0])):
        query_ans = naiveBayes[0][x]
        for y in range(1,len(naiveBayes)):
            query_ans *= naiveBayes[y][theQuery[y-1]][x]
        rootPdf[x] = query_ans
    rootPdf = divide(rootPdf,sum(rootPdf))
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
    jPtemp = jP
    seterr(all = 'ignore')
    jPtemp = jPtemp/jP.sum(axis=1)[:,None]
    jPtemp = jPtemp/jP.sum(axis=0)[None,:]
    jPtemp[jPtemp == 0] = 1
    jPtemp[jP == 0] = 1
    mi = sum(jP*log2(jPtemp))
# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for i in range(noVariables):
        for j in range(noVariables):
            MIMatrix[i][j] = MutualInformation(JPT(theData,j,i,noStates))
            #if i ==j:
            #    MIMatrix[i][j] = 0
# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    a = depMatrix.max()
    while a != 0:
        i,j = where(a==depMatrix)
        i = int(i[0])
        j = int(j[0])
        if j<i:
            j,i = i,j
        depList.append([a,i,j])
        depMatrix[i][j],depMatrix[j][i] = 0,0
        a = depMatrix.max()
# end of coursework 2 task 3
    return array(depList)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    rows,cols = depList.shape
    connections = zeros((noVariables, noVariables), float)
    for i in range(rows):
        x = depList[i][1]
        y = depList[i][2]
        if connections[x][y]==0:
            spanningTree.append(depList[i])
            connections[x][y],connections[y][x] = 1,1
            for k in range(noVariables):
                links =[]
                for l in range(noVariables):
                    if connections[k][l] == 1:
                        links.append(l)
                for n in range(len(links)):
                    for m in range(n+1, len(links)):
                        connections[links[n]][links[m]], connections[links[m]][links[n]] = 1,1

    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from the data set
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
##    for y in unique(theData[:,parent1]):
##        for z in unique(theData[:,parent2]):
##            #print y
##            #print z
##            index = intersect1d(argwhere(theData[0:,parent1] == y), argwhere(theData[0:,parent2] == z))
##            val = asarray(theData[index,child].flatten().tolist())
##            if val.any():
##                cptcol = bincount(val,minlength=noStates[child]).astype(float)
##                cptcol = divide(cptcol,sum(cptcol))
##                cPT[:,y,z] = cptcol
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
def HepCBayesianNetwork(theData, noStates):
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
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here
    for i in cptList:
        sh  = i.shape
        sh_sum = sh[0] - 1
        for j in sh[1:]:
            sh_sum= sh_sum* j
        mdlSize += sh_sum
    mdlSize = mdlSize * log2(noDataPoints) / 2.0
# Coursework 3 task 3 ends here
    return mdlSize
#

#######################
def is_number(s):
    try:
        float(s)
        return True
    except:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except:
        pass
    return False

def len_calc(x):
    if is_number(x):
        return 1
    else:
        return len(x)


#######################


# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
    for i in range(len(dataPoint)):
        j = 1
        temp_cpt = cptList[i][dataPoint[i]]
        while len_calc(temp_cpt) >= 2:
            temp_cpt = temp_cpt[dataPoint[arcList[i][j]]]
            j += 1
        jP = jP*temp_cpt
# Coursework 3 task 4 ends here
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
    for i in theData:
        #print mdlAccuracy
        mdlAccuracy = mdlAccuracy + log2(JointProbability(i, arcList, cptList))
# Coursework 3 task 5 ends here
    return mdlAccuracy
#


def BestScoreNetwork(theData, arcList, cptList, noStates, noDataPoints):
    first = 1
    for j in arcList:
        cptList_temp = copy.deepcopy(cptList)
        arcList_temp = copy.deepcopy(arcList)
        cptList_temp2 = copy.deepcopy(cptList)
        arcList_temp2 = copy.deepcopy(arcList)
        if len(j) == 2:
            cptList_temp[j[0]] = Prior(theData, j[0], noStates)
            arcList_temp[j[0]] = [j[0]]
            score_temp = MDLSize(arcList_temp, cptList_temp, noDataPoints, noStates) - MDLAccuracy(theData, arcList_temp, cptList_temp)
            if first == 1:
                best_score = score_temp
                cpt_best = cptList_temp
                arc_best = arcList_temp
                first = 0
            else:
                if score_temp < best_score:
                    best_score = score_temp
                    cpt_best = cptList_temp
                    arc_best = arcList_temp
        elif len(j) == 3:
            cptList_temp[j[0]] = CPT(theData, j[0], j[1], noStates)
            arcList_temp[j[0]] = [j[0], j[1]]
            score_temp = MDLSize(arcList_temp, cptList_temp, noDataPoints, noStates) - MDLAccuracy(theData, arcList_temp, cptList_temp)
            cptList_temp2[j[0]] = CPT(theData, j[0], j[2], noStates)
            arcList_temp2[j[0]] = [j[0], j[2]]
            score_temp2 = MDLSize(arcList_temp2, cptList_temp2, noDataPoints, noStates)  - MDLAccuracy(theData, arcList_temp2, cptList_temp2)
            if first == 1:
                best_score = score_temp
                cpt_best = cptList_temp
                arc_best = arcList_temp
                first = 0
                if score_temp2 < best_score:
                    best_score = score_temp2
                    cpt_best = cptList_temp2
                    arc_best = arcList_temp2
            else:
                if score_temp < best_score:
                    best_score = score_temp
                    cpt_best = cptList_temp
                    arc_best = arcList_temp
                if score_temp2 < best_score:
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
    mean = realData.mean(axis=0)

    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here
    covar = cov(realData.T)

    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    # Coursework 4 task 3 begins here
    i = 0
    for im in theBasis:
        filename = "PrincipalComponent"+ str(i) + ".jpg"
        SaveEigenface(im,filename)
        i = i + 1
    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = zeros(len(theBasis))
    # Coursework 4 task 4 begins here
    for i in range(len(theBasis)):
        magnitudes[i] = dot(subtract(theFaceImage,theMean),theBasis[i])
        print "magnitudes[%d]:%f" % (i,magnitudes[i])
        print "theFaceImage \tmax: %f ---\t min: %f"   % (max(theFaceImage),min(theFaceImage))
        print "theMean \tmax: %f ---\t min: %f"   % (max(theMean),min(theMean))
        print "theBasis \tmax: %f ---- \t min: %f" % (max(theBasis[i]),min(theBasis[i]))
        # magnitudes[i] = dot(theBasis[i].T,subtract(theFaceImage,theMean))
    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    # Coursework 4 task 5 begins here
    for i in range(0,len(aBasis)+1):
        bias = zeros(len(aBasis))
        bias[0:i] = ones(i)
        # print max(aMean)
        # print max(dot(bias*componentMags,aBasis))
        # print min(aMean)
        # print min(dot(bias*componentMags,aBasis))
        im =  dot(bias*componentMags,aBasis) + aMean
        filename = "PartialReconstruction"+ str(i) + ".jpg"
        SaveEigenface(im,filename)
        #AppendList(filename,im)
    # Coursework 4 task 5 ends here

# def PrincipalComponents(theData):
#     orthoPhi = []
#     # Coursework 4 task 3 begins here
#     # The first part is almost identical to the above Covariance function, but because the
#     # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
#     # The output should be a list of the principal components normalised and sorted in descending
#     # order of their eignevalues magnitudes
#     U = array(theData)
#     mean_u = U.mean(axis=0)
#     #std_u = std(U, axis=0)
#     U = subtract(U,mean_u)
#     #U = divide(U,std_u)
#     theMean = mean_u
#     SaveEigenface(theMean, "MyMean.jpg")
#     uut = dot(U,U.T)

#     eigval, eigvec = linalg.eig(uut)
#     # print "eigvec shapes"
#     # print eigvec.shape
#     # print "U shape:"
#     # print U.shape
#     utu_eig = dot(U.T, eigvec)
#     utu_eig = divide(utu_eig,sqrt(eigval).T)
#     orthoPhi = divide(utu_eig,sqrt(len(theData)-1)).T
#     # Coursework 4 task 6 ends here
#     return array(orthoPhi), theMean

def PrincipalComponents(theData):
    theData = array(theData)
    mu = mean(theData, axis = 0, dtype=float64)
    U = array(theData - mu)
    UUT = dot(U, transpose(U))
    eigvals, eigvecs = linalg.eig(UUT)
    UTU_eigvecs = dot(transpose(U), eigvecs.T)
    # UTU_eigvecs = dot(transpose(U), eigvecs.T)
    rows, cols = UTU_eigvecs.shape
    #UTU_eigvecs[:,j] = UTU_eigvecs[:,j]/sqrt((len(theData)-1)*eigvals[j])
    for j in range(cols):
        UTU_eigvecs[:,j] = UTU_eigvecs[:,j]/(linalg.norm(UTU_eigvecs[:,j]))
    
    orthoPhi = UTU_eigvecs.T
    return orthoPhi, mu
#
# main program part for Coursework 1
#
# noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
# theData = array(datain)
# AppendString("results.txt","Coursework One Results by sg3510")
# AppendString("results.txt","") #blank line
# AppendString("results.txt","The prior probability of node 0")
# prior = Prior(theData, 0, noStates)
# AppendList("results.txt", prior)
# #
# # continue as described
# #
# #

# cpt = CPT(theData,2,0,noStates)
# AppendArray("results.txt", cpt)

# jpt = JPT(theData,2,0,noStates)
# AppendArray("results.txt", jpt)

# cpt_2 = JPT2CPT(jpt)
# AppendArray("results.txt", cpt_2)

# naiveBayes = [prior, CPT(theData,1,0,noStates), CPT(theData,2,0,noStates), CPT(theData,3,0,noStates), CPT(theData,4,0,noStates), CPT(theData,5,0,noStates)]
# query_ans = Query([4,0,0,0,5],naiveBayes)
# AppendList("results.txt", query_ans)
# query_ans = Query([6, 5, 2, 5, 5],naiveBayes)
# AppendList("results.txt", query_ans)

# main program part for Coursework 2
#
##noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
##theData = array(datain)
AppendString("results.txt","Coursework Four Results by sg3510 and mm5110")
##arcList, cptList = HepCBayesianNetwork(theData,noStates)
##print cptList
##mdl_size = MDLSize(arcList, cptList, noDataPoints, noStates)
##AppendString("results.txt","%f" % (mdl_size))
##mdl_acc = MDLAccuracy(theData, arcList, cptList)
##AppendString("results.txt","%f" % (mdl_acc))
##AppendString("results.txt","%f" % (mdl_size - mdl_acc))
##
##cpt_best, arc_best, best_score = BestScoreNetwork(theData, arcList, cptList, noStates, noDataPoints)
##AppendString("results.txt","%f" % (best_score))

#####################
# noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
# theData = array(datain)
# hpcmean = Mean(theData)
# AppendList("results.txt", hpcmean)
# cova = Covariance(theData)
# AppendArray("results.txt", cova)
# theBasis = ReadEigenfaceBasis()

# CreateEigenfaceFiles(theBasis)
# theMean = ReadOneImage("MeanImage.jpg")
# print "minmax"
# print max(theMean)
# print min(theMean)
# theFaceImage = ReadOneImage("c.pgm")
# weights = ProjectFace(theBasis,theMean, theFaceImage)
# # AppendList("results.txt", weights)

# CreatePartialReconstructions(theBasis,theMean, weights)
#####################
##################################################################################################################
theData = ReadImages()
theBasis, theMean = PrincipalComponents(theData)
print theBasis
CreateEigenfaceFiles(theBasis)
# theMean = ReadOneImage("MyMean.jpg")
theFaceImage = ReadOneImage("c.pgm")
weights = ProjectFace(theBasis,theMean, theFaceImage)
CreatePartialReconstructions(theBasis,theMean, weights)
##################################################################################################################




# jpt = JPT(theData,2,0,noStates)
# data = array([[0.3, 0.2],[0.20, 0.30]])
# #print MutualInformation(data);
# depm =  DependencyMatrix(theData, noVariables, noStates)
# depl =  DependencyList(depm)
# set_printoptions(precision=2)
# print "Dependency Matrix"
# print DependencyMatrix(theData, noVariables, noStates)
# print "Dependency List"
# print depl
# print "Spanning Tree"
# print SpanningTreeAlgorithm(depl,noVariables)