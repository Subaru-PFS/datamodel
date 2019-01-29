class MaskHelper:
    """Helper for dealing with symbolic names for mask values

    For science, we care about the symbolic name (i.e., what the mask
    represents; e.g., ``NO_DATA``), but this needs to be translated to the
    implementation (i.e., an integer) so pixels can be selected.

    Parameters
    ----------
    **kwargs : `dict` mapping `str` to `int`
        The mask planes. The integers should all be positive in the range 0..63.
    """
    maskPlanePrefix = "MP_"  # Prefix for header keywords

    def __init__(self, **kwargs):
        self.flags = kwargs
        assert all(ii >= 0 and ii < 64 and isinstance(ii, int) for ii in kwargs.values())

    def __repr__(self):
        """Representation"""
        return "%s(%s)" % (self.__class__.__name__, self.flags)

    def __iter__(self):
        """Iterator"""
        return iter(self.flags)

    def __getitem__(self, name):
        """Retrieve value for a single mask name"""
        return 2**self.flags[name]

    def get(self, *args):
        """Retrieve value for multiple masks"""
        return sum(self[name] for name in args)

    @classmethod
    def fromFitsHeader(cls, header):
        """Read from a FITS header

        Parameters
        ----------
        header : `dict`
            FITS header keyword-value pairs.

        Returns
        -------
        self : `MaskHelper`
            Constructed mask helper.
        """
        maskPlanes = {}
        for key, value in header.items():
            if key.startswith(cls.maskPlanePrefix) or key.startswith("HIERARCH " + cls.maskPlanePrefix):
                name = key[key.rfind(cls.maskPlanePrefix) + len(cls.maskPlanePrefix):]
                maskPlanes[name] = value
                continue
        return cls(**maskPlanes)

    def toFitsHeader(self):
        """Write to a FITS header

        Returns
        -------
        header : `dict`
            FITS header keyword-value pairs.
        """
        return {self.maskPlanePrefix + key: value for key, value in self.flags.items()}

    @classmethod
    def fromMerge(cls, helpers):
        """Construct from multiple `MaskHelper`s

        There must be no discrepancies between the inputs.

        Parameters
        ----------
        helpers : iterable of `MaskHelper`
            `MaskHelper`s to merge.

        Returns
        -------
        self : `MaskHelper`
            Merged `MaskHelper`.
        """
        maskPlanes = {}
        for hh in helpers:
            for name, value in hh.flags.items():
                if name in maskPlanes:
                    if maskPlanes[name] != value:
                        raise RuntimeError("Cannot merge MaskHelpers due to mismatch: %s" % (name,))
                else:
                    maskPlanes[name] = value
        return cls(**maskPlanes)
