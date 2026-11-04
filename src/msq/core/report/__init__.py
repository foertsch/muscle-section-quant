"""The report base shared by the three assays' HTML pages: filling a template (one data marker, shared assets inlined),
images as data URIs, fiber outlines as SVG paths, histogram bins and the choice of the section a page shows."""

from msq.core.report.render import render

__all__ = ["render"]
