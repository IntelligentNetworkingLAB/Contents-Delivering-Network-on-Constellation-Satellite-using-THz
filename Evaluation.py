import math
import numpy as np 
import random
from MyConstants import *
import torch 
import onnx 
import onnxruntime
from onnx import shape_inference, numpy_helper
from dijkstar import Graph, find_path


TOTAL_LATENCY = 0

def InitiallizeEnvrionment(matrix, powerRange):
    if type(powerRange) != type(()):
        print("Power Range must be tuple")
        return
    
    for rowIdx, row in enumerate(matrix):
        for colIdx, elem in enumerate(row):
            matrix[rowIdx][colIdx] = random.randrange(powerRange[0], powerRange[1])

    return matrix

def CalcPathLoss(distance):
    pathLoss = 10 * math.log2(math.pow((4*math.pi*distance * BANDWIDTH / C), 2))
    return pathLoss

def CalcLatency(agentPower, distance):
    squareAE = math.pow(ANTENNA_EFFICIENCY, 2)
    squareAG = math.pow(ANTENNA_GAIN, 2)

    logUpper = agentPower * squareAE * squareAG
    logLower = ADDITIONAL_NOISE_Receiver * BOLTZMAN * ENVTEMP * BANDWIDTH * CalcPathLoss(distance)

    lower = BANDWIDTH * math.log2(1 + (logUpper/logLower))

    latency = DATA_AMOUNT / lower
    return latency

def CalcEnergy(agentPower, distance):
    squareAE = math.pow(ANTENNA_EFFICIENCY, 2)
    squareAG = math.pow(ANTENNA_GAIN, 2)
   
    power = agentPower + (agentPower * squareAE * squareAG) / CalcPathLoss(distance)
    latency = CalcLatency(agentPower, distance)

    return power * latency

def IsConstraint(agentPower):
    result = True

    curEnergy = CalcEnergy(agentPower, FIXED_DISTANCE)
    EMAX = agentPower * LMAX
    if(agentPower > PMAX):
        result = False
    
    if(curEnergy > EMAX):
        result = False
    
    if(TOTAL_LATENCY > TOTAL_LMAX):
        result = False

    return result

def dijkstraMapping(matrix):
    rowSize = len(matrix)
    size = rowSize ** 2

    graph = Graph()

    for idx in range(size):
        row = idx // rowSize
        col = idx % rowSize
        # print("ROW COL ", row, col)

        if (col != rowSize-1): # 가장 오른쪽 라인 제외 오른쪽 노드와 연결 
            graph.add_edge(idx, row*rowSize + col + 1, matrix[row][col])
            if(row != rowSize - 1): # 제일 아래 줄 은 가질 수 없어 
                graph.add_edge(idx, (row+1)*rowSize + col+1, matrix[row][col])
            if(row != 0): # 가장 윗줄 제외 오른쪽 위 연결 
                graph.add_edge(idx, (row-1)*rowSize + col+1, matrix[row][col])

        if (col != 0): # 가장 왼쪽 라인 제외 왼쪽 노드 와 연결 
            graph.add_edge(idx, row*rowSize + col - 1, matrix[row][col])

            if(row != rowSize - 1): # 왼쪽 아래 노드와 엣지 추가 
                graph.add_edge(idx, (row+1)*rowSize + col-1, matrix[row][col])
            if(row != 0): # 왼쪽 위 노드와 엣지 추가 
                graph.add_edge(idx, (row-1)*rowSize + col-1, matrix[row][col])

        if(row != rowSize-1): # 가장 아래 라인 제외 아래 노드와 연결 
            graph.add_edge(idx, (row+1)*rowSize + col, matrix[row][col])
        if (row != 0):
            graph.add_edge(idx, (row-1)*rowSize + col, matrix[row][col])
    return graph


if __name__ == "__main__":
    
    satelliteMatrix = []
    maxStep = 100
    for row in range(50):
        tmpRow = []
        for col in range(50):
            tmpRow.append(0)    
        satelliteMatrix.append(tmpRow)

    satelliteMatrix = InitiallizeEnvrionment(satelliteMatrix, (1, 11))
    satelliteGraph = dijkstraMapping(satelliteMatrix)
    # for satellites in satelliteMatrix:
    #     print(satellites)

    onnx_path = "./20220929_3/MyBehavior.onnx"
    session = onnxruntime.InferenceSession(onnx_path) # Session Input : [maxStep - currentStep, currentAgentPmax, totalEpisodeLatency]

    currentStep = 0
    agentPower = 0

    # outputs[2][0][j]

    path = find_path(satelliteGraph, 0, 24)
    print(path)

    total_power = 0
    curPowerList = []
    curEnergyList = []

    curPowerList_random = []
    curEnergyList_random = []
    totalLatency_random = 0

    curPowerList_min = []
    curEnergyList_min = []
    totalLatency_min = 0

    for idx, agentIndex in enumerate(path.nodes):
        currentStep = idx

        # agent 위치 찾는 중 
        rowSize = 50
        agentRow = agentIndex // rowSize
        agentCol = agentIndex % rowSize
        
        # agent max power 
        agentMaxPower = satelliteMatrix[agentRow][agentCol]

        randomPower = random.randrange(1, agentMaxPower+1)

        # print("CURRENT STEP : ", currentStep)
        # print(idx, "' s MAX POWER IS : ", agentMaxPower)
        inputs_nt = [maxStep - currentStep, agentMaxPower, TOTAL_LMAX - TOTAL_LATENCY]
        inputs = [inputs_nt]    
        outputs = session.run(None, {'obs_0': inputs})

        # print(outputs)
        # for output in outputs:
        #     print(output)
        prediction = torch.clamp(torch.Tensor(outputs[2][0]), -1.0, 1.0)
        
        # 여기서 사용한 power 나오면 저장 + Latency 저장
        currentPower = 5 * prediction + 5.01
        curPowerList.append(currentPower)
        # print("CURRENT POWER : ", currentPower)

        total_power = total_power + currentPower
        currentLatency = CalcLatency(currentPower, FIXED_DISTANCE)
        currentEnergy = CalcEnergy(currentPower, FIXED_DISTANCE)
        curEnergyList.append(currentEnergy[0])

        currentLatency_random = CalcLatency(randomPower, FIXED_DISTANCE)
        currentEnergy_random = CalcEnergy(randomPower, FIXED_DISTANCE)
        curEnergyList_random.append(currentEnergy_random)
        totalLatency_random = totalLatency_random + currentLatency_random
        
        currentLatency_min = CalcLatency(0.01, FIXED_DISTANCE)
        currentEnergy_min = CalcEnergy(0.01, FIXED_DISTANCE)
        curEnergyList_min.append(currentEnergy_min)
        totalLatency_min = totalLatency_min + currentLatency_min


        TOTAL_LATENCY = TOTAL_LATENCY + currentLatency

    print("====================REAL=========================")
    print("Current Power List : ", curPowerList)
    print("Current Energy List : ", curEnergyList)
    print("TOTAL ENERGY : ", sum(curEnergyList))
    print("TOTAL LATENCY : ", TOTAL_LATENCY)
    print("=============================================")

    print("====================RANDOM=========================")
    print("Current Power List : ", curPowerList_random)
    print("Current Energy List : ", curEnergyList_random)
    print("TOTAL ENERGY : ", sum(curEnergyList_random))
    print("TOTAL LATENCY : ", totalLatency_random)
    print("=============================================")

    print("====================minimum=========================")
    print("Current Power List : ", curPowerList_min)
    print("Current Energy List : ", curEnergyList_min)
    print("TOTAL ENERGY : ", sum(curEnergyList_min))
    print("TOTAL LATENCY : ", totalLatency_min)
    print("=============================================")
