import sys
import math
import os

if len(sys.argv)!=7 :
    print("invalid input parameters, usage is:");
    print("python "+sys.argv[0]+" superclass inputFile outputfilePositives outputfileNegatives filenamesPrefix backgroundsFolder")
    exit(-1)

try:
    superclass = int(sys.argv[1])
except ValueError:
    print("invalid superclass (first argument)")
    exit(-2)
inputFileName = sys.argv[2]
outputFileNamePositives = sys.argv[3]
outputFileNameNegatives = sys.argv[4]
filenamesPrefix = sys.argv[5]
backgroundsPath = sys.argv[6]


print("\n")
with open(inputFileName) as input,  open(outputFileNamePositives,"w") as output, open(outputFileNameNegatives,"w") as outputNeg:
    for line in input:
        words = line.split(";")
        if words[0].split("/")[0]=="01" and int(words[6])==superclass:
            #print("words = "+str(words))
            x1 = int(math.floor(float(words[1])))
            y1 = int(math.floor(float(words[2])))
            x2 = int(math.floor(float(words[3])))
            y2 = int(math.floor(float(words[4])))

            #print("x1 = "+str(x1))
            #print("y1 = "+str(y1))
            #print("x2 = "+str(x2))
            #print("y2 = "+str(y2))


            xmin = min(x1,x2)
            xmax = max(x1,x2)
            ymin = min(y1,y2)
            ymax = max(y1,y2)
            height = ymax-ymin
            width = xmax-xmin
            output.write(filenamesPrefix+"/"+words[0]+" "+str(1)+" "+str(xmin)+" "+str(ymin)+" "+str(width)+" "+str(height)+"\n")
        elif words[0].split("/")[0]=="01":
            outputNeg.write(filenamesPrefix+"/"+words[0]+"\n")

with open(outputFileNameNegatives,"a") as outputNeg:
    for file in os.listdir(backgroundsPath):
        outputNeg.write(backgroundsPath+"/"+file+"\n")
