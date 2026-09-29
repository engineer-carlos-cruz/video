"""Errors raised by the pipeline."""


class PipelineError(Exception):
    """Base class for every error raised by the pipeline."""


class StepOutputError(PipelineError):
    """A step failed or produced no usable output.

    The orchestrator stops the flow as soon as this is raised, so every step
    must wrap its failures here instead of returning ``None``.
    """


class UnsupportedURLError(StepOutputError):
    """The URL does not belong to a source supported by step 1."""
