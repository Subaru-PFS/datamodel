#!/usr/bin/env python

from argparse import ArgumentParser
from glob import glob
from typing import Iterable
from pfs.datamodel.pfsConfig import checkPfsConfigHeader
from astropy import log


def main(filenames: Iterable[str], allowFix: bool = False):
    """Main driver to check/fix pfsConfig headers

    Parameters
    ----------
    filenames : Iterable[str]
        List of files (or globs) for which to check/fix headers.
    allowFix : bool, optional
        Allow headers to be fixed?
    """
    numGood = 0
    numBad = 0
    for gg in filenames:
        for fn in glob(gg):
            try:
                checkPfsConfigHeader(fn, allowFix, log)
                numGood += 1
            except Exception as exc:
                log.warning(exc)
                numBad += 1
    log.info(f"Processed {numGood} good and {numBad} bad files.")


if __name__ == "__main__":
    parser = ArgumentParser(description="""
We used to rely on the file names of pfsConfig files in order to determine
important data (pfsDesignId, visit). This practise is fragile, but many files
were written with this scheme. We now write that data to the header so that we
are not restricted to a particular filename template. This script checks that
the header keywords (W_PFDSGN, W_VISIT) are present in the header, and if not
then it writes them based on the values derived from the filename.
""")
    parser.add_argument("--fix", default=False, action="store_true", help="Fix bad/missing headers?")
    parser.add_argument("filenames", nargs="+", help="Names of files (or glob) to check")
    args = parser.parse_args()
    main(args.filenames, args.fix)
