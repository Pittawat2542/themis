def test_interfaces_module_documents_abc_vs_protocol_rationale():
    """The interfaces module docstring must explain the ABC vs Protocol choice."""
    import themis.interfaces as iface

    docstring = iface.__doc__ or ""
    assert "ABC" in docstring or "abstract" in docstring.lower()
    assert "Protocol" in docstring or "protocol" in docstring.lower()
