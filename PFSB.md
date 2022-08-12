PFS H4 ramp files (`PFSB` files)
====

The PFS H4 images are saved in `PFSB` files, each of which contain all of the data from an entire ramp. The detectors are currently always read up-the-ramp and each read is saved in an HDU with an `EXTNAME` of `IMAGE_%d` (1-indexed). The detectors are almost always configured with Interleaved Reference Pixels ("IRP"); those pixels are de-interleaved from the data pixels and saved in their own `REF_%d` HDU, saved right after the associated `IMAGE_%d` HDU. If the system is configured to *not* use IRP, a 0-valued `REF_%d` HDU is saved.

All `PFSB` files written after 2022-02-01 will always start with a pair of `RESET_IMAGE_%d` and `RESET_REF_%d` HDUs, which record an essentially instantaneous per-pixel read made immediately after that pixel is reset at the start of the ramp. We do not expect to take or save multiple reset frames per ramp: for now the reset HDU index will always be 1. As with the `REF_%d` HDUs, if IRP is not used, a 0-valued image will be saved.

In summary, the HDUs in a `PFSB` file start with, in order:
- the PHDU, which does not contain image data.
- `RESET_IMAGE_1`, which contains the data pixels for the reset frame.
- `RESET_REF_1`, which contains the reference pixels for the reset frame, or zeros if no IRP.

After that the `PFSB` files contain N pairs of HDUs:
- `IMAGE_%d`: the data pixels from the Nth read in the ramp
- `REF_%d`: the reference pixels from the Nth read in the ramp, or zeros if no IRP.

-----

The PHDU contains some published H4-specific cards:

- `W_H4FFMT`: the version of the FITS file format. As of 2022-02-01, this is version 3 and is described by this document. The revision list will be described below.
- `W_FRMTIM`: the time required for the ASIC to cycle a single data frame (a complete data+IRP read).
- `W_H4NRED`: the number of reads requested for this ramp.
- `W_H4NRST`: the number of resets requested for this ramp. Will almost certainly always be 1.
- `W_H4NCHN`: the number of channels the ASIC and ROIC are configured for. For PFS, almost always 32, but can also be 16, 4, or 1.
- `W_H4IRP`: T if we are using and reading Interleaved Reference Pixels.
- `W_H4IRPN`: if `W_H4IRP`, the ratio of data to reference pixels. 1..N
- `W_H4IRPO`: if `W_H4IRP`, the 1-based index of the reference pixel within the `W_H4IRPN` data pixels. Usually N, but could be 1 or (N+1)//2.
- `W_H4GAIN`: the ASIC preamp gain. This is the only part of the e^-/ADU gain which is controllable in the ASIC.

The standard `EXPTIME` card is simply `W_H4NRED` * `W_FRMTIM`.

The `IMAGE` HDU headers have a couple of interesting cards:
- `INHERIT=T`. Useful for readers which support that, like the FPS DRP.
- `W_H4READ`. The 1-based index of the read within the ramp.
- `W_H4GRUP`. The 1-based read group within the ramp. PFS does not forsee this being anything other than 1, but it might be for some engineering data.

The values of a few changing cards (e.g. telescope axes, rotator, dome temperature) will be added to the `IMAGE` headers, but most will simply `INHERIT` from the PHDU.

The `REF` HDUs have nothing in their header: they are simply associated data for the `IMAGE_%d` HDUs.

FITS Format versions, per `W_H4FFMT`
----

The `W_H4FFMT` card indicates structural changes to the PFSB files, and not changes just to the headers.

- 1 (indicated by a missing `W_H4FFMT` card): The detector images were swapped L-to-R.
- 2: The detector image readout order was corrected, per [INSTRM-1349](https://pfspipe.ipmu.jp/jira/browse/INSTRM-1349).
- 3: added `RESET_IMAGE_%d` and `RESET_REF_%d` HDUs per [INSTRM-1336](https://pfspipe.ipmu.jp/jira/browse/INSTRM-1336).

Notes
----

All images are saved as 16-bit integer pixels, with BZERO and BSCALE set to support reading as unsigned integers.

The image HDUs may or may not be RICE-compressed. As of 2022-02-01 they will be. Before then they usually were not ([INSTRM-1111](https://pfspipe.ipmu.jp/jira/browse/INSTRM-1336)).

Single-image `PFSA` files were generated for the H4s for a while. They contained a single reference corrected, last-read minus first-read summary image. They could never be correct and were easy to mistake for valid raw data, so we stopped generating them. 

We save the `REF` HDUs even when not using IRP so that readers do not need to special case non-IRP data. We save a full-sized frame of 0-valued pixels instead of an empty image (legal!) for the same reason.

Version 3 also supports "row-skipping" images, per [INSTRM-1484](https://pfspipe.ipmu.jp/jira/browse/INSTRM-1484). This does not affect the structure of the file, but it should be noted that like in windowed CCD reads the FITS images are full-sized, also with 0-valued pixels in the unread area. 