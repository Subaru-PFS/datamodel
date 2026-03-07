class VersionTable:
    """Fixed mapping between version keys and FITS header cards."""

    _tables = {}

    @classmethod
    def register(cls, name, table):
        """Register a table.

        Parameters
        ----------
        name : `str`
            Logical name (e.g. "design", "config0", "config").
        table : iterable of (`str`, `str`, `str`)
            Entries (keyName, headerCard, comment).
        """
        cls._tables[name] = tuple(table)

    @classmethod
    def _getTable(cls, name):
        """Return a registered table or raise with a helpful message."""
        try:
            return cls._tables[name]
        except KeyError as exc:
            valid = ", ".join(sorted(cls._tables))
            raise KeyError(f"Unknown VersionTable '{name}'. Valid: {valid}") from exc

    @classmethod
    def read(cls, header, name):
        """Read a versions dict from header using a registered table."""
        table = cls._getTable(name)
        versions = {}
        for keyName, card, _ in table:
            value = header.get(card)
            if value:
                versions[keyName] = str(value)
        return versions

    @classmethod
    def write(cls, header, name, versions):
        """Write a versions dict to header using a registered table.

        This clears all cards in the table first to avoid stale values persisting
        across multiple writes.
        """
        table = cls._getTable(name)

        for _, card, _ in table:
            if card in header:
                del header[card]

        for keyName, card, comment in table:
            value = versions.get(keyName, None)
            if value:
                header[card] = (str(value), comment)


tableDesign = (
    ("pfs_instdata", "W_DSVR00", "pfs_instdata version used to create PfsDesign"),
    ("pfs_utils", "W_DSVR01", "pfs_utils version used to create PfsDesign"),
    ("datamodel", "W_DSVR02", "pfs.datamodel version used to create PfsDesign"),
    ("author", "W_DSVR03", "author that created PfsDesign"),
    ("ets_fiberalloc", "W_DSVR04", "ets_fiberalloc version used to create PfsDesign"),
    ("ets_pointing", "W_DSVR05", "ets_pointing version used to create PfsDesign"),
    ("ets_shuffle", "W_DSVR06", "ets_shuffle version used to create PfsDesign"),
    ("ets_target_database", "W_DSVR07", "ets_target_database version used to create PfsDesign"),
    ("ics_cobraCharmer", "W_DSVR08", "ics_cobraCharmer version used to create PfsDesign"),
    ("ics_cobraOps", "W_DSVR09", "ics_cobraOps version used to create PfsDesign"),
    ("ics_fpsActor", "W_DSVR10", "ics_fpsActor version used to create PfsDesign"),
    ("pfs_obsproc_planning", "W_DSVR11", "pfs_obsproc_planning version used to create PfsDesign"),
    ("qplan", "W_DSVR12", "qplan version used to create PfsDesign"),
    ("spt_operational_database", "W_DSVR13", "spt_operational_database version used to create PfsDesign"),
)

tableConfig0 = (
    ("pfs_instdata", "W_C0VR00", "pfs_instdata version used for convergence"),
    ("pfs_utils", "W_C0VR01", "pfs_utils version used for convergence"),
    ("datamodel", "W_C0VR02", "pfs.datamodel version used for convergence"),
    ("author", "W_C0VR03", "author that created pfsConfig0"),
    ("ics_cobraCharmer", "W_C0VR04", "ics_cobraCharmer version used for convergence"),
    ("ics_fpsActor", "W_C0VR05", "ics_fpsActor version used for convergence"),
    ("ics_mcsActor", "W_C0VR06", "ics_mcsActor version used for convergence"),
    ("moduleXml", "W_C0VR07", "XML version used for convergence"),
)

tableConfig = (
    ("pfs_instdata", "W_CFVR00", "pfs_instdata version used for exposure"),
    ("pfs_utils", "W_CFVR01", "pfs_utils version used for exposure"),
    ("datamodel", "W_CFVR02", "pfs.datamodel version used for exposure"),
    ("author", "W_CFVR03", "author that created pfsConfig"),
    ("ics_iicActor", "W_CFVR04", "ics_iicActor version used for exposure"),
    ("ics_spsActor", "W_CFVR05", "ics_spsActor version used for exposure"),
)

VersionTable.register("design", tableDesign)
VersionTable.register("config0", tableConfig0)
VersionTable.register("config", tableConfig)
