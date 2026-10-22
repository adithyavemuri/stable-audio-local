"""Capture official decoder samples before API peak scaling or clipping."""
class RawDecoderCapture:
    """Wrap one decode method without changing official model computations."""
    def __init__(self, instance, method_name):
        self.tensor = None
        original = getattr(instance, method_name)

        def capture(*args, **kwargs):
            output = original(*args, **kwargs)
            self.tensor = output.detach().float().cpu().clone()
            return output

        setattr(instance, method_name, capture)

    def take(self, frames):
        """Return the requested sample window, retaining original amplitude."""
        if self.tensor is None:
            raise RuntimeError("The raw decoder hook did not run")
        result = self.tensor[0, :, :frames]
        self.tensor = None
        if result.shape[-1] != frames:
            raise RuntimeError("Raw decoder length differs from output window")
        return result
