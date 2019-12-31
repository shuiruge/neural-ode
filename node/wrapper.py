from node.core import get_node_function


def node_wrapper(solver, t0):
    """Returns a decorator."""
    return lambda fn: get_node_function(solver, t0, fn)


def get_node_method(solver, t0, method):

    def node_method(obj, t, x):

        def fn(t, x):
            return method(obj, t, x)

        node_fn = get_node_function(solver, t0, fn)
        return node_fn(t, x)

    return node_method


def node_method_wrapper(solver, t0):
    """Returns a decorator."""
    return lambda method: get_node_method(solver, t0, method)
