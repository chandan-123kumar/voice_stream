from .talker_megakernel import TalkerMegakernel  # noqa: F401

__all__ = ["TalkerMegakernel", "MegakernelTTS"]


def __getattr__(name):
    if name == "MegakernelTTS":
        from .engine import MegakernelTTS
        return MegakernelTTS
    raise AttributeError(name)
