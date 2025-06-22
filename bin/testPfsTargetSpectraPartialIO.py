#!/usr/bin/env python

import os
from time import time
import resource
from argparse import ArgumentParser
from pfs.datamodel import PfsCalibrated


def get_bytes_read(pid):
    """Return the number of bytes read so far for the given process ID."""

    with open(f"/proc/{pid}/io", "r") as f:
        for line in f:
            if line.startswith("rchar:"):
                return int(line.split()[1])

    return 0


def main(filename, **kwargs):
    # Remove None values from kwargs
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Get the PID of the current process
    pid = os.getpid()

    usage = resource.getrusage(resource.RUSAGE_SELF)
    bytes_start = get_bytes_read(pid)
    time_start = time()
    print('Bytes read so far:', bytes_start)
    print('Max RSS:', usage.ru_maxrss, 'kB')
    print('Minor page faults so far:', usage.ru_minflt)
    print('Major page faults so far:', usage.ru_majflt)

    # Load a subset of the spectra
    spectra = PfsCalibrated.readFits(filename, **kwargs)
    print(f"Loaded {len(spectra)} spectra with kwargs: {kwargs}")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    bytes_partial = get_bytes_read(pid)
    time_partial = time()
    print('Bytes read during loading:', bytes_partial - bytes_start)
    print('Time taken to load a subset of spectra:', time_partial - time_start)
    print('Max RSS:', usage.ru_maxrss, 'kB')
    print('Minor page faults so far:', usage.ru_minflt)
    print('Major page faults so far:', usage.ru_majflt)

    # Load the entire file
    spectra = PfsCalibrated.readFits(filename)
    print(f"Loaded {len(spectra)} spectra.")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    bytes_all = get_bytes_read(pid)
    time_all = time()
    print('Bytes read during loading:', bytes_all - bytes_partial)
    print('Time taken to load all spectra:', time_all - time_partial)
    print('Max RSS:', usage.ru_maxrss, 'kB')
    print('Minor page faults so far:', usage.ru_minflt)
    print('Major page faults so far:', usage.ru_majflt)


if __name__ == "__main__":
    parser = ArgumentParser(description="""
Transitioning to gen3 Butler resulted in combining target spectra into large FITS files which
contain many PfsSingle or PfsObject spectra. This script can be used to benchmark the IO cost
saved by partially loading PfsCalibrated or PfsCoadd spectra, rather than loading the
entire file.
""")
    parser.add_argument("filename", help="Name of the file to load.")
    parser.add_argument("--targetId", type=int, help="Target IDs to load (default: all).")
    parser.add_argument("--targetType", type=int, help="TargetType to load (default: all).")
    parser.add_argument("--catId", type=int, help="Objects with catId to load (default: all).")
    args = parser.parse_args()
    main(args.filename,
         targetId=args.targetId)
