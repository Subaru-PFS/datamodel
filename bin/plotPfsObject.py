#!/usr/bin/env python

from pfs.datamodel.drp import PfsObject


def plotPfsObject(dirName, catId, tract, patch, objId, nVisit, pfsVisitHash):
    """Plot a PfsObject from disk

    Parameters
    ----------
    dirName : `str`
        Name of directory containing the data.
    catId : `int`
        Catalog identifier.
    tractId : `int`
        Tract identifier.
    patch : `str`
        Patch identifier (typically two integers separated by a comma).
    objId : `int`
        Object identifier.
    nVisit : `int`
        Number of visits contributing.
    pfsVisitHash : `int`
        Hash of contributing visits.

    Returns
    -------
    figure : `matplotlib.Figure`
        Figure containing the plot.
    axes : `matplotlib.Axes`
        Axes containing the plot.
    """
    identity = dict(catId=catId, tract=tract, patch=patch, objId=objId,
                    nVisit=nVisit, pfsVisitHash=pfsVisitHash)
    obj = PfsObject.read(identity, dirName=dirName)
    return obj.plot(show=False)


def integer(xx):
    """Create an integer using auto base detection"""
    return int(xx, 0)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Plot PfsObject")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing data")
    parser.add_argument("--catId", type=int, required=True, help="Catalog identifier")
    parser.add_argument("--tract", type=int, required=True, help="Tract")
    parser.add_argument("--patch", type=str, required=True, help="Patch")
    parser.add_argument("--objId", type=integer, required=True, help="Object identifier")
    parser.add_argument("--nVisit", type=int, required=True, help="Number of visits")
    parser.add_argument("--pfsVisitHash", type=integer, required=True, help="Hash of contributing visits")

    args = parser.parse_args()
    import matplotlib.pyplot as plt
    fig, axes = plotPfsObject(args.dir, args.catId, args.tract, args.patch, args.objId,
                              args.nVisit, args.pfsVisitHash)
    plt.show()  # Keeps the plot open
    return fig, axes


if __name__ == "__main__":
    main()
