"""Shared helpers for the plotting modules."""

import matplotlib.pyplot as plt


def save_pdf(save_path):
    """
    Save the current figure as a vector PDF with a transparent background. Fonts are embedded as
    TrueType (Type 42) instead of matplotlib's default Type 3, so text stays editable in vector
    editors and the PDF is accepted by journals that reject Type 3 fonts.
    """
    with plt.rc_context({"pdf.fonttype": 42}):
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=True)
