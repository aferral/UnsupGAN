from launchers.run_exp import train
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("configFile", help="Where is the configFile")
parser.add_argument("--train", help="Do training",action="store_true")
parser.add_argument("--classExp", help="Do clasifyExp",action="store_true")
parser.add_argument("--discr", help="Do discriminator exp",action="store_true")
parser.add_argument("--modeCollapse", help="Do modeCollapse exp",action="store_true")

args = parser.parse_args()

configFilePath = args.configFile
print "About to use ",configFilePath
if args.train:
    print "train turned on"
    train(configFilePath)
if args.classExp:
    print "clasifyExp turned on"
    os.system("python -m launchers.clasifyExp"+ " " + configFilePath)
if args.discr:
    print "discriminator turned on"
    os.system("python -m launchers.discriminatorTest"+ " " + configFilePath)
if args.modeCollapse:
    print "modeCollapse turned on"
    os.system("python -m launchers.modeCollapseTest"+ " " + configFilePath)













