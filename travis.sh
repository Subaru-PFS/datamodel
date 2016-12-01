#!/bin/bash

#
# Script to initiate testing on Travis-CI
#
# Testing is done through the 'pfs_pipe2d' package, which needs to be downloaded.
#

set -ev

PIPE2D_URL=git://github.com/Subaru-PFS/pfs_pipe2d

BUILD_BRANCH=XXX
if [ -n "$TRAVIS_PULL_REQUEST_BRANCH" ]; then
    BUILD_BRANCH=$TRAVIS_PULL_REQUEST_BRANCH
else
    BUILD_BRANCH=$TRAVIS_BRANCH
fi
echo "Building branch $BUILD_BRANCH ..."

cd $HOME
git clone $PIPE2D_URL
cd $(basename $PIPE2D_URL)
git checkout $BUILD_BRANCH || echo "Cannot checkout branch $BUILD_BRANCH; this is not fatal"
./bin/pfs_travis.sh
