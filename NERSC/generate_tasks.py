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

queueType = "premium"
maxTime = 12  # in hours
repeatRange = range(0, 10, 10)

taskRange=range(0,5)

##### End of input parameters #####

##### Checking that right directories exist and creating them as needed #####

confFileNames = [baseDir+"/tasks_"+str(i)+".txt" for i in taskRange]

confFiles = []

for i in taskRange:
    confFile = open(confFileNames[i], "w")
    confFile.writelines("#!/usr/bin/env bash\n\n")
    confFiles.append(confFile)

subFileNames = [baseDir+"/sub_to_run_"+str(i)+".sl" for i in taskRange]

totalTasks=math.ceil((repeatRange.stop-repeatRange.start)/repeatRange.step*50.0)
jobPerTask=math.ceil(totalTasks/(taskRange.stop-taskRange.start))
nodePerTask=jobPerTask+1
#jobPerTask=nodePerTask*64

for i in taskRange:
    subFile = open(subFileNames[i], "w")
    subFile.writelines(
        '''#!/usr/bin/env bash

#SBATCH -q {queueType}
#SBATCH --constraint=haswell

#SBATCH -N {nodePerTask}
#SBATCH --license cscratch1

#SBATCH --job-name=analysis
#SBATCH --time 4:00:00
#SBATCH --image=docker:wenkaifan/jetscape_analyzer:latest

runcommands.sh tasks_{i}.txt'''.format(queueType=queueType, nodePerTask=nodePerTask, i=i))

##### End of directory checking/creation #####

count=0
for j in range(0,50):

    outputDir= baseDir+"/design_jetscape/main/"+str(j)+"/"
    intputDir= baseDir.replace("/Analysis", "/Simulation")+"/design_jetscape/main/"+str(j)+"/"+"OutputFiles/"
    for m in repeatRange:
        i=math.floor(count*(taskRange.stop-taskRange.start)/totalTasks)
        confFiles[i].writelines(baseDir+"/JetScapeAnalyzer/NERSC/work_leading.sh "+inputDir+" "+outputDir+" "+m+\n")
        count+=1
