"""The lm-polygraph estimators used by the benchmark (MIT, see
LICENSE-lm-polygraph.md).

Vendored rather than installed so the scores do not depend on which
lm-polygraph release is installed. Compared with the copy this was taken
from, KernelLanguageEntropy builds the Laplacian as D - W and symmetrises E
and C for each question, and SemanticEntropy has extra (unused here)
estimation options. Everything else is unchanged apart from imports.
"""
