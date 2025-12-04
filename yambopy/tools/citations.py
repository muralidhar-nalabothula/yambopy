import atexit

# Global citation registry
_CITATIONS_USED = set()

def citation(ref: str):
    """
    Decorator to track citation usage.

    When the decorated function is executed, the provided citation reference
    is recorded in an internal registry. Citations are printed once on program
    exit.

    Parameters
    ----------
    ref : str
        The textual reference (e.g., paper title, DOI, URL) to record when the
        function is used.

    Returns
    -------
    callable
        Wrapped function that logs its citation on each call.

    Example
    -------
    >>> from  yambopy.tools.citation import citation

    >>> @citation("Ref A: Paper on Algorithm A")
    >>> def function_a():

    >>> #Note that the citation must be same else, duplication can happen.

    """
    def wrapper(func):
        def inner(*args, **kwargs):
            _CITATIONS_USED.add(ref)  # record the citation
            return func(*args, **kwargs)
        inner.__doc__ = func.__doc__
        inner.__name__ = func.__name__
        return inner
    return wrapper


@atexit.register
def print_citations():
    """
    Print a summary of citations used during the program execution.

    If any decorated functions were executed, this function displays a list
    of unique references in sorted order when Python terminates.

    Notes
    -----
    This function is automatically registered with :mod:`atexit` and
    should not be called manually in normal operation.
    """
    if _CITATIONS_USED:
        print("\n=========================================================")
        print("Please cite the following references in case you wish")
        print("to acknowledge the work done by the contibutors.")
        print("=========================================================")
        for i, ref in enumerate(sorted(_CITATIONS_USED), start=1):
            print(f"{i}) {ref}")
        print("=========================================================")


#### test
if __name__ == "__main__":
    @citation("Ref A: Tenet by C. Nolan")
    def Tenet():
        return 1 + 1

    @citation("Ref B: Intersteller by C. Nolan")
    def Intersteller():
        return 1 + 1
    # call functcion
    for i in range (3) :
        Intersteller()
        Tenet()
