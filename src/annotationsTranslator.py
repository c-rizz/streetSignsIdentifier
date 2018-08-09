 #!/usr/bin/env python3

#This is a script to build opencv_traincascade annotations from the BelgiumTS and German traffic sign datasets

# IMPORTANT:
# This source contains non-ascii characters, so use python 3 (or just run this as a script as there's the shebang)




import sys
import math
import os
import csv
from random import shuffle

if len(sys.argv)<5 :
    print("invalid input parameters, usage is:");
    print("python "+sys.argv[0]+" inputFile outputfilePositives outputfileNegatives backgroundsFolder germanDatasetFolder  [options] [--super <superclass> or --classes <(class,class,class,...)>]")
    print(" that is, yu should specify a superclass or a list of classes")
    exit(-1)


inputFileName = sys.argv[1]
outputFileNamePositives = sys.argv[2]
outputFileNameNegatives = sys.argv[3]
backgroundsPath = sys.argv[4]
germanPath = sys.argv[5]

optionstart = 6
option = sys.argv[optionstart]
if option=="--maxGerman":
    maxGerman = int(sys.argv[optionstart+1])
    optionstart+=2

option = sys.argv[optionstart]
superclasses = []
classes = []
if option=="--super":
    rest = ""
    for s in sys.argv[optionstart+1:]:
        rest = rest + s
    print("rest = "+rest)
    superclasses = list(map(int, rest.split(",")))
elif option=="--classes":
    rest = ""
    for s in sys.argv[optionstart+1:]:
        rest = rest + s
    if rest[0]!="("  or rest.find(")")!=len(rest)-1:
        print("invalid classes syntax, should be: --class (class1,class2,...)")
        exit(-2)
    rest = rest[1,len(rest)-1]
    classes = res.split(",")
"""
classes:
0.    other defined TS = [all the other defined TS ids besides the following 11]; %0
1.    triangles = [2 3 4 7 8 9 10 12 13 15 17 18 22 26 27 28 29 34 35];   %1 (corresponds to Danger superclass in GTSDB)
2.    redcircles  = [36 43 48 50 55 56 57 58 59 61 65]; %2 (corresponds to Prohibitory superclass in GTSDB)
3.    bluecircles = [72 75 76 78 79 80 81];    %3 (corresponds to Mandatory superclass in GTSDB)
4.    redbluecircles = [82 84 85 86];    %4
5.    diamonds = [32 41]; %5
6.    revtriangle = [31]; %6
7.    stop = [39]; %7
8.    forbidden = [42];%8
9.    squares = [118 151 155 181]; %9
10.    rectanglesup  = [37,87,90,94,95,96,97,149,150,163];%10
11.    rectanglesdown= [111,112]; %11
"""
germanFoldersPerSuperclass = [[],
["00011","00018","00019","00020","00021","00022","00023","00024","00025","00026","00027","00028","00029","00030","00031"],
["00000","00001","00002","00003","00004","00005","00006","00007","00008","00009","00010","00015","00016"],
["00033","00034","00035","00036","00037","00038","00039","00040"],
[],
["00012"],
["00013"],
["00014"],
["00017"],
[],
[],
[]]

poscount = 0
negcount = 0
lineCounter = 0
posCountByCam = [0,0,0,0,0,0,0,0]
with open(inputFileName) as input,  open(outputFileNamePositives,"w") as output, open(outputFileNameNegatives,"w") as outputNeg:
    for line in input:
        words = line.split(";")
        #print("interperting "+str(words))
        try:
            camera = int(words[0].split("/")[0])
            if (int(words[6]) in superclasses) or (words[5] in classes):
                #print("words = "+str(words))
                x1 = int(math.floor(float(words[1])))
                y1 = int(math.floor(float(words[2])))
                x2 = int(math.floor(float(words[3])))
                y2 = int(math.floor(float(words[4])))
                #print("x1"+str(x1)+" y1"+str(y1)+" x2"+str(x2)+" y2"+str(y2))
                xmin = min(x1,x2)
                xmax = max(x1,x2)
                ymin = min(y1,y2)
                ymax = max(y1,y2)
                height = ymax-ymin
                width = xmax-xmin
                #print("xmin"+str(xmin)+" ymin"+str(ymin)+" xmax"+str(xmax)+" ymax"+str(ymax)+" w"+str(width)+" h"+str(height))

                #ensure it is square, this could result in a cropped sample :(
                cx = (xmax+xmin)/2
                cy = (ymax+ymin)/2
                size = max(width,height)
                if int(words[6])==1:
                    size +=6 #to take some more background, especially for the bottom part, hoping we dont go out of the image
                #print("cx"+str(cx)+" cy"+str(cy)+" size"+str(size))
                squareX = int(cx - size/2)
                squareY = int(cy - height/2)
                #print("squareX"+str(squareX)+" squareY"+str(squareY))
                #print("        result: "+words[0]+" "+str(1)+" "+str(squareX)+" "+str(squareY)+" "+str(size)+" "+str(size)+"\n")
                output.write(words[0]+" "+str(1)+" "+str(squareX)+" "+str(squareY)+" "+str(size)+" "+str(size)+"\n")
                poscount+=1
                posCountByCam[camera] += 1
            else:
                outputNeg.write(words[0]+"\n")
                negcount += 1
        except:
            print("invalid syntax at line "+str(lineCounter))
        lineCounter+=1
    germanPos=0
    germanRows =[]
    for superclass in superclasses:
        for germanFolder in germanFoldersPerSuperclass[superclass]:
            if germanPath[len(germanPath)-1]!= "/":
                germanPath+="/"
            with open(germanPath+germanFolder+"/GT-"+germanFolder+".csv", newline='') as csvfile:
                file = csv.reader(csvfile, delimiter=';', quotechar='|')
                next(file)#skip header
                for row in file:
                    x1 = int(math.floor(float(row[3])))
                    y1 = int(math.floor(float(row[4])))
                    x2 = int(math.floor(float(row[5])))
                    y2 = int(math.floor(float(row[6])))
                    #print("x1"+str(x1)+" y1"+str(y1)+" x2"+str(x2)+" y2"+str(y2))
                    xmin = min(x1,x2)
                    xmax = max(x1,x2)
                    ymin = min(y1,y2)
                    ymax = max(y1,y2)
                    height = ymax-ymin
                    width = xmax-xmin
                    #print("xmin"+str(xmin)+" ymin"+str(ymin)+" xmax"+str(xmax)+" ymax"+str(ymax)+" w"+str(width)+" h"+str(height))

                    #ensure it is square, this could go outside the image D:, hope for the best
                    cx = (xmax+xmin)/2
                    cy = (ymax+ymin)/2
                    size = max(width,height)
                    #if superclass==1:
                    #    size +=6 #to take some more background, especially for the bottom part, hoping we dont go out of the image
                    #print("cx"+str(cx)+" cy"+str(cy)+" size"+str(size))
                    squareX = int(cx - size/2)
                    squareY = int(cy - height/2)
                    #print("squareX"+str(squareX)+" squareY"+str(squareY))
                    #print("        result: "+row[0]+" "+str(1)+" "+str(squareX)+" "+str(squareY)+" "+str(size)+" "+str(size)+"\n")
                    germanRows.append(germanPath+"/"+germanFolder+"/"+row[0]+" "+str(1)+" "+str(squareX)+" "+str(squareY)+" "+str(size)+" "+str(size)+"\n")
    shuffle(germanRows)
    numGerman = min(maxGerman,len(germanRows))
    for gRow in germanRows[0:numGerman]:
        output.write(gRow)
        germanPos+=1
        poscount+=1


with open(outputFileNameNegatives,"a") as outputNeg:
    for file in os.listdir(backgroundsPath):
        outputNeg.write(backgroundsPath+"/"+file+"\n")
        negcount += 1


#randomize the order of the elements in the files, son numpos and numNeg dont take the images in the order of the datasets (which have similar images one after the other)
import random
with open(outputFileNamePositives,'r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open(outputFileNamePositives,'w') as target:
    for _, line in data:
        target.write( line )


with open(outputFileNameNegatives,'r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open(outputFileNameNegatives,'w') as target:
    for _, line in data:
        target.write( line )






print("The file paths in the generated files are for a folder structure like:")
print("./")
print("├── 00")
print("│   ├── image.000001.jp2")
print("|   └── ...")
print("├── 01")
print("│   ├── image.000002.jp2")
print("|   └── ...")
print("├── ...")
print("├── "+germanPath)
print("│   ├── 00000")
print("│   |   ├── 00000_00001.ppm")
print("|   |   └── ...")
print("|   └── ...")
print("├── "+backgroundsPath)
print("│   ├── image.000003.jp2")
print("|   └── ...")
print("├── "+outputFileNamePositives)
print("└── "+outputFileNameNegatives)

print("positives count =  "+str(poscount))
i  = 0
for c in posCountByCam:
    print("   "+str(c)+" from camera "+str(i))
    i+=1
print("   "+str(germanPos)+" from the german dataset")
print("negatives count = "+str(negcount))
