def test_import_core_packages():
    # 基本包导入 smoke test，用于快速验证安装与 PYTHONPATH 配置是否正确
    import vulcan  # noqa: F401
    import vulcan.framework  # noqa: F401
    import vulcan.lang  # noqa: F401
    import vulcan.datacollection  # noqa: F401
    import vulcan.cli  # noqa: F401
    import vulcan.services  # noqa: F401

