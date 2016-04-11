import matplotlib.pyplot as plt

from pfs.datamodel.pfsConfig import PfsConfig
from pfs.datamodel.pfsArm import PfsArmSet
from pfs.datamodel.pfsObject import PfsObject, makePfsObject

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main(pfsConfigId, tract, patch, dataDir=".",
         showPfsArm=False, showPfsArmSet=False, showPfsObject=False):

    pfsConfig = PfsConfig(pfsConfigId)
    pfsConfig.read(dataDir)

    objId = pfsConfig.objId[0]

    print "objId = 0x%x" % (objId)

    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    pfsArms = PfsArmSet(visits=1, spectrograph=1)
    pfsArms.read(dataDir)

    if showPfsArm:
        pfsArms.data[1]['r'].plot(showFlux=True, showSky=True, showCovar=True, showMask=True)

    if showPfsArmSet:
        pfsArms.plot(showFlux=True, showSky=True, showCovar=True, showMask=True)

    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    pfsObject = makePfsObject(tract, patch, objId, pfsArms)
    pfsObject.write(dataDir)

    npfs = PfsObject(tract, patch, objId, visits=[1])
    npfs.read(dataDir)
    if showPfsObject:
        npfs.plot(showFlux=True, showSky=True, showCovar=True, showCovar2=True)
        npfs.plot(showFluxTbl=True, showFluxVariance=False)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import argparse
parser = argparse.ArgumentParser(description="Manipulate pfsConfig, pfsArm, and pfsObject files")

parser.add_argument('pfsConfigId', type=str, nargs="?", default="0x741918352327a27",
                    help="Desired pfsConfigId")
parser.add_argument('--dataDir', type=str, default="examples", help="Directory containing data")
parser.add_argument('--tract', type=int, default=0, help="Desired tract")
parser.add_argument('--patch', type=str, default="0,0", help="Desired patch")
parser.add_argument('--showPfsArm', action="store_true", help="Plot an pfsArm file?")
parser.add_argument('--showPfsArmSet', action="store_true", help="Plot set of pfsArm files?")
parser.add_argument('--showPfsObject', action="store_true", help="Plot pfsObject file?")

args = parser.parse_args()

main(int(args.pfsConfigId, 16), tract=args.tract, patch=args.patch, dataDir=args.dataDir,
     showPfsArm=args.showPfsArm, showPfsArmSet=args.showPfsArmSet,
     showPfsObject=args.showPfsObject)
