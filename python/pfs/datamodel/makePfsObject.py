import collections
from scipy.interpolate import interp1d
from pfs.datamodel.pfsArm import PfsArm


NCOARSE = 10


def makePfsObject(objId, pfsArms, catId=0, lambdaMin=350, lambdaMax=1260, dLambda=0.1):
    """Create a PfsObject from a list of PfsArmSet objects"""
    visits = [aset.visit for aset in pfsArms]
    #
    # Check that all the pfsArmSets are consistent
    #
    pfsConfig = None
    for aset in pfsArms:
        for arm in aset.data.values():
            fiberIdx = np.where(arm.pfsConfig.objId == objId)[0]
            if len(fiberIdx):
                fiberIdx = fiberIdx[0]
            else:
                raise RuntimeError("Object 0x%x is not present in pfsArm file for "
                                   "visit %d, spectrograph %d%s" %
                                   (objId, aset.visit, arm.spectrograph, arm.arm))

        if pfsConfig:
            assert (pfsConfig.tract, pfsConfig.patch) == (aset.pfsConfig.tract, aset.pfsConfig.patch)
        else:
            pfsConfig = aset.pfsConfig
            tract = pfsConfig.tract[fiberIdx]
            patch = pfsConfig.patch[fiberIdx]

    if False:
        pfsVisitHash = calculate_pfsVisitHash(visits)  # noqa: F841 (never used)
    pfsObject = PfsObject(tract, patch, objId, catId, visits=visits)
    pfsObject.pfsConfigIds = []         # we'll set them from the pfsArm files

    pfsObject.lam = lambdaMin + dLambda*np.arange(int((lambdaMax - lambdaMin)/dLambda), dtype=np.float32)
    #
    # Start by interpolating all the data onto the single uniform sampling
    #
    armFlux = {}
    armVariance = {}
    armSky = {}
    armMask = {}

    for aset in pfsArms:
        visit = aset.visit
        armFlux[visit] = {}
        armVariance[visit] = {}
        armMask[visit] = {}
        armSky[visit] = {}

        for arm in aset.data.values():
            pfsObject.pfsConfigIds.append(arm.pfsConfigId)
            fiberIdx = np.where(arm.pfsConfig.objId == objId)[0]
            if len(fiberIdx):
                fiberIdx = fiberIdx[0]
            else:
                raise RuntimeError("Object 0x%x is not present in pfsArm file for "
                                   "visit %d, spectrograph %d%s" %
                                   (objId, visit, arm.spectrograph, arm.arm))
            #
            # how to interpolate
            #
            kwargs = dict(kind='linear',
                          bounds_error=False,
                          fill_value=0,
                          copy=True,
                          # assume_sorted=True
                          )

            armFlux[visit][arm.arm] = interp1d(arm.lam[fiberIdx], arm.flux[fiberIdx],
                                               **kwargs)(pfsObject.lam)
            armSky[visit][arm.arm] = interp1d(arm.lam[fiberIdx], arm.sky[fiberIdx],
                                              **kwargs)(pfsObject.lam)
            kwargs.update(fill_value=np.inf)
            armVariance[visit][arm.arm] = interp1d(arm.lam[fiberIdx], arm.covar[fiberIdx][0],
                                                   **kwargs)(pfsObject.lam)
            kwargs.update(fill_value=PfsArm.flags["NODATA"], kind='nearest')
            armMask[visit][arm.arm] = interp1d(arm.lam[fiberIdx], arm.mask[fiberIdx],
                                               **kwargs)(pfsObject.lam)
            armMask[visit][arm.arm] = armMask[visit][arm.arm].astype(arm.mask[fiberIdx].dtype)

    pfsObject.flux = np.zeros_like(pfsObject.lam)
    pfsObject.covar = np.zeros(3*len(pfsObject.lam)).reshape(3, len(pfsObject.lam))
    pfsObject.mask = np.zeros_like(pfsObject.lam, dtype=np.int32)
    pfsObject.sky = np.zeros_like(pfsObject.lam)

    weights = np.zeros_like(pfsObject.lam)
    #
    # Average together all the arm data
    #
    for aset in pfsArms:
        visit = aset.visit
        for arm in aset.data:
            w = 1/armVariance[visit][arm]
            pfsObject.flux += w*armFlux[visit][arm]
            pfsObject.sky += w*armSky[visit][arm]
            pfsObject.mask |= np.where(w > 0, armMask[visit][arm], 0)
            weights += w

    noData = (weights == 0)
    pfsObject.flux /= np.where(noData, 1, weights)
    pfsObject.flux[noData] = np.nan

    pfsObject.sky /= np.where(noData, 1, weights)
    pfsObject.sky[noData] = np.nan

    pfsObject.mask[noData] = PfsArm.flags["NODATA"]

    with np.errstate(divide='ignore'):
        pfsObject.covar[0] = np.where(weights == 0, np.inf, 1/weights)
        pfsObject.covar[1] = np.where(weights == 0, np.inf, 0)
        pfsObject.covar[2] = np.where(weights == 0, np.inf, 0)
    #
    # Set the coarse-grained covariance
    #
    # For now, we'll just set it from the diagonal part of covar, binned into NCOARSE pieces
    #
    C = np.zeros((pfsObject.NCOARSE, pfsObject.NCOARSE))
    pfsObject.covar2 = C

    binsize = len(pfsObject.lam)//pfsObject.NCOARSE
    idx = (np.arange(len(pfsObject.lam))//binsize).astype(int)

    goodData = np.logical_not(noData)
    for j in range(pfsObject.NCOARSE):
        x = pfsObject.covar[0][np.logical_and(goodData, idx == j)]
        C[j, j] = x.sum()/np.sqrt(len(x))
    #
    # Now the best-estimate flux at the native resolution of the pfsArm files
    #
    overlaps = []
    arms = pfsArms[0].data
    if 'b' in arms and 'r' in arms:
        overlaps.append(('b', 'r'))
    if 'r' in arms and 'n' in arms:
        overlaps.append(('r', 'n'))

    fluxTbl = collections.OrderedDict()
    fiberIdx = np.where(list(arms.values())[0].pfsConfig.objId == objId)[0][0]  # we checked existence above

    if not overlaps:
        for armStr in arms:
            fluxTbl[armStr] = PfsObject.FluxTbl(arms[armStr].lam[fiberIdx])
    else:
        #
        # how to interpolate
        #
        kwargs = dict(kind='linear',
                      bounds_error=False,
                      copy=True,
                      # assume_sorted=True
                      )
        firstLam = None                  # the first wavelength to include from previous overlap
        for a1, a2 in overlaps:
            var2 = interp1d(arms[a2].lam[fiberIdx], arms[a2].covar[fiberIdx][0],
                            fill_value=np.inf, **kwargs)(arms[a1].lam[fiberIdx])
            #
            # Set an array to the values where the covariance is less in a1 than a2
            #
            # Because the covariances are a bit noisy where the arms cross, and
            # because the interpolation giving var2 can smooth things, use all
            # the points to the left of the last acceptable value
            useA1 = arms[a1].covar[fiberIdx][0] < var2
            lastGoodIdx = np.argwhere(useA1)[-1][-1]
            useA1 = np.where(np.arange(len(useA1)) <= lastGoodIdx, True, False)
            #
            # Remember the first wavelength that we should use from the next overlap
            #
            if firstLam:
                useA1 = np.logical_and(useA1, arms[a1].lam[fiberIdx] >= firstLam)
                firstLam = None

            fluxTbl[a1] = PfsObject.FluxTbl(arms[a1].lam[fiberIdx][useA1])
            #
            # set firstLam for the next overlap
            #
            if lastGoodIdx < len(useA1) - 1:   # we want at least one value from a2
                firstLam = arms[a1].lam[fiberIdx][lastGoodIdx + 1]
        #
        # include the last arm
        #
        a1 = overlaps[-1][1]
        useA1 = np.ones_like(useA1)
        if firstLam:
            useA1 = np.logical_and(useA1, arms[a1].lam[fiberIdx] >= firstLam)
            firstLam = None

        fluxTbl[a1] = PfsObject.FluxTbl(arms[a1].lam[fiberIdx][useA1])
    #
    # OK, we have the range of wavelengths that we want to use,
    # so interpolate the data onto those wavelengths (arm by arm) and average
    #
    assert len(fluxTbl) == len(pfsArms[0].data)
    for armStr in fluxTbl:
        armFlux = {}
        armVariance = {}
        armMask = {}

        for aset in pfsArms:
            arm = aset.data[armStr]
            fiberIdx = np.where(arm.pfsConfig.objId == objId)[0][0]  # we checked existence above
            #
            # how to interpolate
            #
            kwargs = dict(kind='linear',
                          bounds_error=False,
                          copy=True,
                          # assume_sorted=True
                          )

            armFlux[arm] = interp1d(arm.lam[fiberIdx], arm.flux[fiberIdx],
                                    fill_value=0, **kwargs)(fluxTbl[armStr].lam)
            armVariance[arm] = interp1d(arm.lam[fiberIdx], arm.covar[fiberIdx][0],
                                        fill_value=np.inf, **kwargs)(fluxTbl[armStr].lam)
            kwargs.update(fill_value=PfsArm.flags["NODATA"], kind='nearest')
            armMask[arm] = interp1d(arm.lam[fiberIdx], arm.mask[fiberIdx], **kwargs)(fluxTbl[armStr].lam)
            armMask[arm] = armMask[arm].astype(arm.mask[fiberIdx].dtype)

        weights = np.zeros_like(fluxTbl[armStr].lam)
        #
        # Average together all the arm data
        #
        for aset in pfsArms:
            visit = aset.visit
            w = 1/armVariance[arm]
            fluxTbl[armStr].flux += w*armFlux[arm]
            fluxTbl[armStr].mask |= np.where(w > 0, armMask[arm], 0)

            weights += w

        noData = (weights == 0)
        fluxTbl[armStr].flux /= np.where(noData, 1, weights)
        fluxTbl[armStr].flux[noData] = np.nan

        with np.errstate(divide='ignore'):
            fluxTbl[armStr].fluxVariance = np.where(weights == 0, np.inf, 1/weights)
    #
    # The PfsObject.FluxTbl isn't actually quite what we need, as it's N separate FluxTbl
    # objects rather than one covering all the arms.  Fix this
    #
    nPoint = np.sum(len(ft.flux) for ft in fluxTbl.values())
    lam = np.empty(nPoint, dtype=np.float32)
    pfsObject.fluxTbl = PfsObject.FluxTbl(lam)

    i0 = 0
    for ft in fluxTbl.values():
        i1 = i0 + len(ft.lam)
        pfsObject.fluxTbl.lam[i0:i1] = ft.lam
        pfsObject.fluxTbl.flux[i0:i1] = ft.flux
        pfsObject.fluxTbl.mask[i0:i1] = ft.mask
        pfsObject.fluxTbl.fluxVariance[i0:i1] = ft.fluxVariance
        i0 = i1

    return pfsObject
