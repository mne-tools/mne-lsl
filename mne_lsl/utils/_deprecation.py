from functools import wraps
from inspect import Parameter, signature

from .logs import warn


def deprecate_positional_args(f):  # pragma: no cover
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.
    Modified from sklearn utils.validation.

    Parameters
    ----------
    f : Callable
        Function to check arguments on.
    """  # noqa: D401
    sig = signature(f)
    kwonly_args = []
    all_args = []

    for name, param in sig.parameters.items():
        if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            all_args.append(name)
        elif param.kind == Parameter.KEYWORD_ONLY:
            kwonly_args.append(name)

    @wraps(f)
    def inner_f(*args, **kwargs):
        extra_args = len(args) - len(all_args)
        if extra_args > 0:
            # ignore first 'self' argument for instance methods
            args_msg = [
                f"{name}"
                for name, _ in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            warn(
                f"Pass '{', '.join(args_msg)}' as keyword arguments. Passing these as "
                "positional arguments will be considered an error in 1.5.",
                FutureWarning,
            )
        for k, arg in zip(sig.parameters, args):
            kwargs[k] = arg
        return f(**kwargs)

    return inner_f
