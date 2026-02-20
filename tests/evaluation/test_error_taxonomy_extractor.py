from themis.evaluation import extractors


def test_error_taxonomy_arithmetic_slip_detected():
    ext = extractors.ErrorTaxonomyExtractor()
    labels = ext.extract("Answer: 2 + 2 = 5")
    assert labels["arithmetic_slip"] is True


def test_error_taxonomy_format_parse_failure_detected():
    ext = extractors.ErrorTaxonomyExtractor()
    labels = ext.extract('{"answer": "Paris"')  # unbalanced brace
    assert labels["format_parse_failure"] is True


def test_error_taxonomy_reasoning_gap_detected():
    ext = extractors.ErrorTaxonomyExtractor()
    labels = ext.extract("Answer: Paris")
    assert labels["reasoning_gap"] is True
