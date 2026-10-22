"""User-facing exceptions for PELBC software."""


class PELBCError(Exception):
    """Base class for expected input, model, and prediction failures."""


class InputValidationError(PELBCError):
    """Raised when an encounter directory or audio clip is incompatible."""


class ModelValidationError(PELBCError):
    """Raised when a supplied or bundled model artifact is missing or malformed."""
