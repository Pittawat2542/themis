"""Tests verifying backward compatibility after export module split."""


class TestExportBackwardCompatibility:
    """All existing import paths must still work after split."""

    def test_import_from_package(self):
        from themis.experiment.export import (
            export_report_csv,
            export_html_report,
            export_report_json,
            export_summary_json,
            export_report_bundle,
            render_html_report,
            build_json_report,
        )

        assert callable(export_report_csv)
        assert callable(export_html_report)
        assert callable(export_report_json)
        assert callable(export_summary_json)
        assert callable(export_report_bundle)
        assert callable(render_html_report)
        assert callable(build_json_report)

    def test_import_submodules(self):
        from themis.experiment.export.csv import export_report_csv

        assert callable(export_report_csv)
