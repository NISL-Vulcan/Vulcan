"""Unit tests for vulcan.datacollection."""
def test_datacollection_import():
    import vulcan.datacollection
    assert hasattr(vulcan.datacollection, "__name__")
