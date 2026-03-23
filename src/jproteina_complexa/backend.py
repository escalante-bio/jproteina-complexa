"""from_torch singledispatch and AbstractFromTorch base class.

The from_torch machinery is only usable when torch is installed.
All other jproteina_complexa modules import from_torch but never call it at runtime,
so torch is not a runtime dependency.
"""

import dataclasses
import functools

import equinox as eqx
import jax.numpy as jnp


@functools.singledispatch
def from_torch(obj):
    """Convert a PyTorch object to its JAX/Equinox equivalent."""
    raise TypeError(f"No from_torch converter registered for {type(obj)}")


# Register base type converters only if torch is available
try:
    import torch

    @from_torch.register(torch.Tensor)
    def _tensor(t):
        return jnp.array(t.detach().cpu().numpy())

    from_torch.register(int, lambda x: x)
    from_torch.register(float, lambda x: x)
    from_torch.register(bool, lambda x: x)
    from_torch.register(type(None), lambda x: None)
    from_torch.register(tuple, lambda t: tuple(from_torch(v) for v in t))
    from_torch.register(list, lambda lst: [from_torch(v) for v in lst])
    from_torch.register(dict, lambda d: {k: from_torch(v) for k, v in d.items()})
except ImportError:
    pass


class AbstractFromTorch(eqx.Module):
    """Base class providing automatic from_torch conversion by matching field names."""

    @classmethod
    def from_torch(cls, model):
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {}

        for name, child in model.named_children():
            if name not in field_names:
                raise ValueError(
                    f"{cls.__name__}: PyTorch child '{name}' has no matching field. "
                    f"Fields: {field_names}"
                )
            kwargs[name] = from_torch(child)

        for name, param in model.named_parameters(recurse=False):
            if name not in field_names:
                raise ValueError(
                    f"{cls.__name__}: PyTorch param '{name}' has no matching field. "
                    f"Fields: {field_names}"
                )
            kwargs[name] = from_torch(param)

        for name, buf in model.named_buffers(recurse=False):
            if name in field_names:
                kwargs[name] = from_torch(buf)

        for f in dataclasses.fields(cls):
            if f.name not in kwargs:
                if f.default is not dataclasses.MISSING:
                    kwargs[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    kwargs[f.name] = f.default_factory()

        return cls(**kwargs)
