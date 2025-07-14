from collections import Counter

__all__ = ("MaskHelper",)


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
    _maxSize = 64  # Maximum number of bits

    def __init__(self, **kwargs):
        self.flags = kwargs
        assert all(ii >= 0 and ii < self._maxSize and isinstance(ii, int) for ii in kwargs.values())

    def __repr__(self):
        """Representation"""
        return "%s(%s)" % (self.__class__.__name__, self.flags)

    def __iter__(self):
        """Iterator"""
        return iter(self.flags)

    def __getitem__(self, name):
        """Retrieve value for a single mask name"""
        return 2**self.flags[name]

    def __len__(self):
        """Number of bits used"""
        return len(self.flags)

    def __contains__(self, name):
        """Is mask name used?"""
        return name in self.flags

    def get(self, *args):
        """Retrieve value for multiple masks"""
        return sum(self[name] for name in args)

    def copy(self):
        """Return a copy"""
        return type(self)(**self.flags)

    def add(self, name):
        """Add mask name"""
        if name in self.flags:
            return self[name]
        if len(self) >= self._maxSize:
            raise RuntimeError("No bits remaining")
        # Find the lowest available bit
        existing = set(self.flags.values())
        for ii in range(self._maxSize):
            if ii not in existing:
                value = ii
                break
        else:
            raise AssertionError("Something's broken")
        self.flags[name] = value
        return self[name]

    @classmethod
    def fromFitsHeader(cls, header, strip=False):
        """Read from a FITS header

        Parameters
        ----------
        header : `dict`
            FITS header keyword-value pairs. Modified if ``strip=True``.
        strip : `bool`
            Strip keywords that we use from the header?

        Returns
        -------
        self : `MaskHelper`
            Constructed mask helper.
        """
        maskPlanes = {}
        used = []
        for key, value in header.items():
            if key.startswith(cls.maskPlanePrefix) or key.startswith("HIERARCH " + cls.maskPlanePrefix):
                name = key[key.rfind(cls.maskPlanePrefix) + len(cls.maskPlanePrefix):]
                maskPlanes[name] = value
                used.append(key)
                continue
        if strip:
            for key in used:
                del header[key]
        return cls(**maskPlanes)

    def toFitsHeader(self):
        """Write to a FITS header

        Returns
        -------
        header : `dict`
            FITS header keyword-value pairs.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file FOR ALL CLASSES THAT USE THIS.
        header = {self.maskPlanePrefix + key: value for key, value in self.flags.items()}
        header = {("HIERARCH " + key if len(key) > 8 else key): value for key, value in header.items()}
        return header

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

    def interpret(self, value):
        """Interpret a value from the mask

        Breaks down the provided value into the corresponding mask plane names.

        Parameters
        ----------
        value : `int`
            Value to interpret.

        names : `list` of `str`
            List of mask planes that are set in the provided value.
        """
        return [nn for nn, vv in self.flags.items() if (value & 2**vv) != 0]

    def count(self, mask):
        """Return counts of each mask plane

        Parameters
        ----------
        mask : `numpy.ndarray`
            Mask array.

        Returns
        -------
        counts : `dict` (`str`: `int`)
            Counts for each mask plane. An additional result indexed by an empty
            string corresponds to the number of pixels with no mask plane set.
        """
        return {",".join(self.interpret(value)): num for value, num in Counter(mask.flatten()).items()}
