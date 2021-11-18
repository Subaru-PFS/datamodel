install()
{
    if test -z "$SCONSUTILS_DIR"; then
        echo disabling scons because we do not support it.
        mv SConstruct SConstruct-disabled
    fi
    default_install "$@"
}
