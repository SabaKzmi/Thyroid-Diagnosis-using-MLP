import math
import pandas as pd
pd.options.mode.chained_assignment = None
import random as rand

#load data
inputDF = pd.read_excel('thyroidInputs.xlsx', header = None)
inputDF = inputDF.transpose()
targetDF = pd.read_excel('thyroidTargets.xlsx', header = None)
targetDF = targetDF.transpose()
weightsDF = pd.read_excel('weights.xlsx')
middleNeuron = 3

#sigmoid activation functions
def activation(NI):
        return 1 / (1 + (math.e ** ((-1)*NI)))
def activationDerivative(NI):
    return activation(NI) * (1-activation(NI))

#test function
def check(testInput):
    result = ":("

    test_zNI = [0.0] * middleNeuron
    test_z = [0.0] * middleNeuron

    test_yNI = [0.0] * 3
    test_y = [0.0] * 3

    # first layer
    for j in range(middleNeuron):
        test_zNI[j] = weightsDF['vb'][j]
        t = 0
        for v in range(21):
            test_zNI[j] += weightsDF['v{}'.format(v)][j] * testInput[t]
            t += 1
        # activate zNI
        test_z[j] = activation(test_zNI[j])

    # second layer
    for k in range(3):
        test_yNI[k] = weightsDF['wb'][k]
        for j in range(middleNeuron):
            test_yNI[k] += weightsDF['w{}'.format(k)][j] * test_z[j]
        # activate zNI
        test_y[k] = activation(test_yNI[k])
    # activation function
    if test_y[0] > 0.5 > test_y[1] and  0.5 > test_y[2]:
        return 'Normal'
    elif test_y[1] > 0.5 > test_y[0] and 0.5 > test_y[2] :
        return 'Hyperfunction'
    elif test_y[2] > 0.5 > test_y[0] and 0.5 > test_y[1]:
        return 'Subnormal functioning'
    else:
        return result

#training function
def train():

    #variable initialization
    epoch = 0
    alpha = 0.1
    validationTest = 0
    repeated_val = 0

    zNI = [0.0] * middleNeuron
    z = [0.0] * middleNeuron

    yNI = [0.0] * 3
    y = [0.0] * 3

    delta_k = [0.0] * 3
    delta_j = [0.0] * middleNeuron

    # set weights
    for m in range(middleNeuron):
        for v in range(21):
            weightsDF['v{}'.format(v)][m] = rand.uniform(-0.5, 0.5)
        for w in range(3):
            weightsDF['w{}'.format(w)][m] = rand.uniform(-0.5, 0.5)
            weightsDF['wb'][w] = rand.uniform(-0.5, 0.5)
        weightsDF['vb'][m] = rand.uniform(-0.5, 0.5)

    # creat train, validation and test lists (random permutation)
    randomList = rand.sample(range(len(inputDF)), len(inputDF))
    trainList = randomList[2 * (len(randomList)//10):] # 80% for training
    validationList = randomList[:len(randomList)//10] #10% for validation
    testList = randomList[len(randomList)//10:2 * (len(randomList)//10)] #10% for test
    print("randomList length: ", len(randomList), "trainList length: ", len(trainList), "validationList length: ", len(validationList), "testList length: ", len(testList))
    print("train list: ", trainList)
    print("validation list: ", validationList)
    print("test list: ", testList)

    while validationTest < 90 and repeated_val < 5:
        for i in trainList:

            # calculate the net input of Z's
            for j in range(middleNeuron):
                zNI[j] = weightsDF['vb'][j]
                for v in range(21):
                    zNI[j] += weightsDF['v{}'.format(v)][j] * inputDF[v][i]
                # activate zNI
                z[j] = activation(zNI[j])

            # calculate the net input of Y's
            for k in range(3):
                yNI[k] = weightsDF['wb'][k]
                for j in range(middleNeuron):
                    yNI[k] += weightsDF['w{}'.format(k)][j] * z[j]
                # activate zNI
                y[k] = activation(yNI[k])

                # calculate delta_k
                delta_k[k] = (targetDF[k][i] - y[k]) * activationDerivative(yNI[k])

            # calculate delta_j
            for j in range(middleNeuron):
                D = 0
                for k in range(3):
                    D += delta_k[k] * weightsDF['w{}'.format(k)][j]
                delta_j[j] = D * activationDerivative(zNI[j])

            # update weights and bias
            # v (from input to middle layer)
            for j in range(middleNeuron):
                for v in range(21):
                    weightsDF['v{}'.format(v)][j] += (alpha * delta_j[j] * inputDF[v][i])
                weightsDF['vb'][j] += delta_j[j] * alpha
            # w (from middle to output)
            for k in range(3):
                for j in range(middleNeuron):
                    weightsDF['w{}'.format(k)][j] += alpha * delta_k[k] * z[j]
                weightsDF['wb'][k] += delta_k[k] * alpha

        epoch += 1

        #validation
        if epoch % 10 == 0:
            print("epoch: {}".format(epoch))
            val_correctAns = 0
            for v in validationList:
                # test
                res = check(inputDF.loc[v, :].tolist())
                # see if the answer was correct
                if res == 'Normal' and targetDF[0][v] == 1 and targetDF[1][v] == 0 and targetDF[2][v] == 0:
                    val_correctAns += 1
                elif res == 'Hyperfunction' and targetDF[0][v] == 0 and targetDF[1][v] == 1 and targetDF[2][v] == 0:
                    val_correctAns += 1
                elif res == 'Subnormal functioning' and targetDF[0][v] == 0 and targetDF[1][v] == 0 and targetDF[2][v] == 1:
                    val_correctAns += 1
            if validationTest == (val_correctAns / len(validationList)) * 100:
                repeated_val += 1
            else:
                repeated_val = 0
            validationTest = (val_correctAns/len(validationList)) * 100
            print("repeated validation percentage: {}   validation percentage: {}".format(repeated_val, validationTest))

    #save the new weights
    weightsDF.to_excel('weights.xlsx', index=False)

    #accuract check
    acc_correctAns = 0
    accRow = 0
    while accRow < len(testList):
        # test
        res = check(inputDF.loc[accRow, :].tolist())
        # see if the answer was correct
        if res == 'Normal' and targetDF[0][accRow] == 1 and targetDF[1][accRow] == 0 and targetDF[2][accRow] == 0:
            acc_correctAns += 1
        elif res == 'Hyperfunction' and targetDF[0][accRow] == 0 and targetDF[1][accRow] == 1 and targetDF[2][accRow] == 0:
            acc_correctAns += 1
        elif res == 'Subnormal functioning' and targetDF[0][accRow] == 0 and targetDF[1][accRow] == 0 and targetDF[2][accRow] == 1:
            acc_correctAns += 1

        accRow += 1

    print("alpha: ", alpha)
    print('Accuracy: ', (acc_correctAns / len(testList)) * 100)

train()