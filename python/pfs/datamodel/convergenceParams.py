class ConvergenceParams:
    """Fixed mapping between convergence parameter names and FITS header cards.

    Everything fps records about one convergence, written once and read by both the FITS
    header and the opdb ingest, so the file and the database cannot disagree about how a
    visit was converged.

    distanceTolerance is the one a reader cannot do without: GOOD means
    |pfiCenter - pfiNominal| <= that value, and it varies per visit.  The rest is
    provenance -- what was asked for, and how it went.

    Note the two tolerances are different numbers.  requestedTolerance is what the
    convergence loop aimed for per iteration; distanceTolerance is the threshold that
    decides fiberStatus, and is floored well above it.

    Values keep their type through a round trip, unlike VersionTable which stringifies:
    a tolerance is a float and a skipped check is a bool.
    """

    _table = (
        ("targetFallbackInvalid", "W_CVFBI", str,
         "where an invalid target is sent"),
        ("targetFallbackUnassigned", "W_CVFBU", str,
         "where an unassigned cobra is sent"),
        ("distanceTolerance", "W_CVTOL", float,
         "[mm] fiberStatus=GOOD if <= distanceTolerance"),
        ("fiducialCheckSkipped", "W_CVSKF", bool,
         "fiducial interference check was skipped"),
        ("numIterations", "W_CVNIT", int,
         "iterations allocated to the convergence"),
        ("elapsedTime", "W_CVETM", float,
         "[s] elapsed time of the convergence"),
        ("requestedTolerance", "W_CVRTL", float,
         "[mm] tolerance for the convergence loop"),
    )

    @classmethod
    def getFieldNames(cls):
        """Return the declared parameter names.

        Returns
        -------
        `tuple` of `str`
            Names in table order: the only keys `read` reports and `write` records.
        """
        return tuple(keyName for keyName, _, _, _ in cls._table)

    @classmethod
    def read(cls, header):
        """Read the convergence parameters from a FITS header.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            Primary header.

        Returns
        -------
        dict
            Only the keys actually present; absent ones are simply not reported, so a
            file written before a parameter existed does not claim a default it never
            used.
        """
        info = {}
        for keyName, card, cast, _ in cls._table:
            if card in header:
                info[keyName] = cast(header[card])

        return info

    @classmethod
    def write(cls, header, info):
        """Write the convergence parameters to a FITS header.

        Cards in the table are cleared first, so a re-write cannot leave a stale value
        behind.  A key absent from `info` writes no card.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            Primary header; modified.
        info : dict
            Convergence parameters, keyed as in the table.
        """
        for _, card, _, _ in cls._table:
            if card in header:
                del header[card]

        for keyName, card, cast, comment in cls._table:
            if info.get(keyName) is not None:
                header[card] = (cast(info[keyName]), comment)
