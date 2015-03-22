import numpy as np

def shuffle_together(*arrays):
    if len(arrays)==0:
        return arrays

    length = len(arrays[0])
    if not all([len(a)==length for a in arrays]):
        raise ValueError, "Given arrays have different lengths"

    random_indices = np.random.permutation(xrange(length))

    shuffled_arrays = [array[random_indices] for array in arrays]
    return shuffled_arrays


def one_out_of_n(labels, n_labels=None):
    """Given an array of integers, return a matrix with the n-th column 1, rest 0."""
    if n_labels is None:
        """We guess the number of features from the largest label number present"""
        n_labels = 1 + np.max(labels)
    m = np.zeros((len(labels), n_labels))
    for (i, label) in enumerate(labels):
        m[i, label] = 1
    return m


def argmax(array):
    return np.unravel_index(np.argmax(array), array.shape)


def rowsort(array):
    array = np.array(array)
    return array[argrowsort(array)]


def argrowsort(array):
    array = np.array(array)
    return np.lexsort(array.T[::-1])


def make_update_globals(globals):
    """Returns a convenience helper that makes variables in a given dict
       available in the global scope, for easy interactive exploration.
       Pass the interpreter's globals() as argument:
       >>> ug = make_update_globals(globals())
       Then run ug(locals()) in any function to publish its local variables.
    """
    def update_globals(variables, name='run'):
        postfix = 0
        while True:
            prefix = name+'_'+`postfix`+'_'
            g_addition = {
                prefix+key: value
                for (key, value) in variables.items()
            }
            # Unpack dicts, and short lists and tuples of items, so their
            # elements are displayed in Spyder's variable explorer.
            for (key, value) in variables.items():
                g_addition.update(_unpack(value, prefix=prefix+key))
            if any([key in globals for key in g_addition]):
                # number already used, try next number
                postfix += 1
                continue
            else:
                # found an unused number
                break
        globals.update(g_addition)
    return update_globals


def _unpack(value, prefix=''):
    """Recursively unpack lists, tuples and dicts to a flat dict.
       Note: dies horribly when given (sth containing) a recursive list/dict!
    """
    retval = {}
    if type(value) in (list, tuple) and len(value) < 10:
        for (i, item) in enumerate(value):
            selector = '['+`i`+']'
            retval[prefix+selector] = item
            retval.update(_unpack(item, prefix=prefix+selector))
    if type(value) is dict:
        for (k, v) in value.items():
            selector = "['"+k+"']"
            retval[prefix+selector] = v
            retval.update(_unpack(v, prefix=prefix+selector))
    return retval


def test_one_out_of_n():
    l = [0, 2, 1, 1, 3]
    m = one_out_of_n(l)
    assert np.all(m == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    m2 = one_out_of_n(l, n_labels=5)
    assert np.all(m2 == [
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ])


def test_argmax():
    a = np.array([
        [ 3.0, 4.0],
        [ 2.0, 0.2],
        [ 1.0, 0.1],
    ])
    assert argmax(a) == (0,1)


def test_rowsort():
    a = np.array([
        [ 3, 3],
        [ 1, 7],
        [ 2, 4],
        [ 2, 2],
    ])
    assert np.all(rowsort(a) == [
        [ 1, 7],
        [ 2, 2],
        [ 2, 4],
        [ 3, 3],
    ])


def test_argrowsort():
    a = np.array([
        [ 3, 3],
        [ 1, 7],
        [ 2, 4],
        [ 2, 2],
    ])
    assert np.all(argrowsort(a) == [1, 3, 2, 0])
    assert np.all(a[argrowsort(a)] == rowsort(a))


def test__unpack():
    l = ['bla', [3,4], {'a': 1, 'b': [5,6]}, 7]
    d = _unpack(l, prefix='l')
    assert d == {
        'l[0]': 'bla',
        'l[1]': [3,4],
        'l[1][0]': 3,
        'l[1][1]': 4,
        'l[2]': {'a': 1, 'b': [5,6]},
        "l[2]['a']": 1,
        "l[2]['b']": [5,6],
        "l[2]['b'][0]": 5,
        "l[2]['b'][1]": 6,
        'l[3]': 7,
    }
