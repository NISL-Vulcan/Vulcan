def test_import_core_packages():
    # Basic package import smoke test for quick installation/PYTHONPATH validation.
    import vulcan  # noqa: F401
    import vulcan.framework  # noqa: F401
    import vulcan.lang  # noqa: F401
    import vulcan.datacollection  # noqa: F401
    import vulcan.cli  # noqa: F401
    import vulcan.services  # noqa: F401

