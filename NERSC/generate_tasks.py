import os
import stat
import os.path
import re
import math
import shutil
import time
import getopt
import sys
import fileinput
import pathlib

def checkAndBuildDir(checkDir):
    if (not os.path.isdir(checkDir)):
        print("Creating directory \""+checkDir+"\" ...")
        os.mkdir(checkDir)


##### Input parameters #####
baseDir = str(pathlib.Path(__file__).parent.absolute())

queueType = "regular"
maxTime = 48  # in hours
repeatRange = range(0, 1, 1)

taskRange=range(0,2)

##### End of input parameters #####

##### Checking that right directories exist and creating them as needed #####

taskType="heavy_jet"

confFileNames = [baseDir+"/tasks_"+taskType+"_"+str(i)+".txt" for i in taskRange]

confFiles = []

for i in taskRange:
    confFile = open(confFileNames[i], "w")
    confFile.writelines("#!/usr/bin/env bash\n\n")
    confFiles.append(confFile)

subFileNames = [baseDir+"/sub_"+taskType+"_"+str(i)+".sl" for i in taskRange]

totalTasks=math.ceil((repeatRange.stop-repeatRange.start)/repeatRange.step*50.0)
jobPerTask=math.ceil(totalTasks/(taskRange.stop-taskRange.start))*10
nodePerTask=math.ceil(jobPerTask/32)+1

for i in taskRange:
    subFile = open(subFileNames[i], "w")
    subFile.writelines(
        '''#!/usr/bin/env bash

#SBATCH -q {queueType}
#SBATCH --constraint=haswell

#SBATCH -N {nodePerTask}
#SBATCH --license cscratch1

#SBATCH --job-name=analysis
#SBATCH --time {maxTime}:00:00

export THREADS=32

runcommands.sh tasks_{taskType}_{i}.txt'''.format(queueType=queueType, nodePerTask=nodePerTask, maxTime=maxTime, taskType=taskType, i=i))

##### End of directory checking/creation #####

count=0
for j in range(0,50):
    outputDir= baseDir+"/design_jetscape/main/"+str(j)+"/"
    inputDir= baseDir.replace("/Analysis", "/Simulation")+"/design_jetscape/main/"+str(j)+"/"+"OutputFiles/"
    for m in repeatRange:
        outputDir=outputDir+"RUN_"+str(m)+"/"
        os.makedirs(outputDir, exist_ok=True)
        i=math.floor(count*(taskRange.stop-taskRange.start)/totalTasks)
        confFiles[i].writelines("shifter --image=docker:wenkaifan/jetscape_analyzer:latest /global/cscratch1/sd/wf39/Analysis/JetScapeAnalyzer/NERSC/work_"+taskType+".sh "+inputDir+" "+outputDir+" "+str(m)+"\n")
        count+=1