#!/usr/bin/env python

import random
from random import randint


description = """
Generate toy data samples.
"""


def expand(modifiers, initial_value, prev_layer):
    """Produce values for the next layer of latent variables

       For each active bit in the previous layer, the corresponding modifier is
       called to perform its trick on next layer's value.
    """
    next_layer = initial_value
    for (bit, modifier) in zip(prev_layer, modifiers):
        if bit == '1':
            next_layer = modifier(next_layer)
    return next_layer


def str_or(s1, s2):
    """Logical OR of two strings of bits"""
    dimensionality = len(s1)
    assert len(s2) == dimensionality
    result = int(s1, 2) | int(s2, 2)
    return format(result, '0%db' % dimensionality)


def or_with(pattern_generator):
    return lambda pattern: str_or(pattern, pattern_generator())


def or_with_one_of(patterns):
    return or_with(lambda: random.choice(list(patterns)))


def set_bits(indices, bitstring):
    bitlist = list(bitstring)
    for index in indices:
        bitlist[index] = '1'
    return ''.join(bitlist)


def l2_generator():
    """Generate a value for the top layer of latent variables"""
    return random.choice([
        '1000',
        '0100',
        '0010',
        '0001',
    ])


# For each 'class', define which combinations of features can be picked.
l2_to_l1_modifiers = [
    or_with_one_of({
        '000001',
        '000010',
        '000011',
    }),
    or_with_one_of({
        '001101',
        '001100',
        '001001',
        '001000',
        '000101',
        '000100',
    }),
    or_with_one_of({
        '110100',
        '110010',
        '110001',
    }),
    or_with_one_of({
        '110000',
        '101000',
        '100100',
        '100010',
        '100001',
    }),
]


l1_generator = lambda l2: expand(l2_to_l1_modifiers, '0'*6, l2)


# For each 'feature', define which bits patterns it can set in a sample.
l1_to_v_modifiers = [
    # 0: Up to ten bits in the left half
    lambda pattern: set_bits([randint(0,9) for _ in range(10)], pattern),

    # 1: Rightmost four bits (16..19), plus up to three other bits in right half
    lambda pattern: set_bits([16,17,18,19]+[randint(10,15) for _ in range(3)],
                             pattern),

    # 2: Always just these three bits (10..12)
    or_with(lambda: '00000000001110000000'),

    # 3: Up to eight of the even bits
    lambda pattern: set_bits([2*randint(0,9) for _ in range(8)], pattern),

    # 4: Up to eight of the odd bits
    lambda pattern: set_bits([1+2*randint(0,9) for _ in range(8)], pattern),

    # 5: Three or four out of four bits of 13..16
    or_with_one_of({
        '00000000000001111000',
        '00000000000001110000',
        '00000000000001101000',
        '00000000000001011000',
        '00000000000000111000',
    }),
]


v_generator = lambda l1: expand(l1_to_v_modifiers, '0'*20, l1)


layer_expanders = [
    l2_generator,
    l1_generator,
    v_generator
]


def sample(latent_vars=[]):
    """Generate a data sample"""
    latent_vars = list(latent_vars)  # copy the list
    if latent_vars == []:
        latent_vars = [layer_expanders[0]()]

    while len(latent_vars) < len(layer_expanders):
        layer_expander = layer_expanders[len(latent_vars)]
        latent_vars.append(layer_expander(latent_vars[-1]))
    return latent_vars


def toydata(n_samples=1, bits=20, latent_vars=[], seed=None):
    """Generate toy data samples."""
    if seed is not None:
        random.seed(seed)

    return [sample(latent_vars) for _ in xrange(n_samples)]


def parse_args(argv):
    import argparse
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('n_samples', nargs='?', type=int, default=8,
                           help="number of data samples to generate")
    argparser.add_argument('--latent-vars', metavar='VALUES',
                           help="Fix latent variable values, e.g. 0001,001010")
    argparser.add_argument('--seed', type=int, help="randomiser seed")
    args = vars(argparser.parse_args(argv))
    # Remove None values to not override function's defaults
    args = {k:v for (k,v) in args.items() if v is not None}
    if 'latent_vars' in args:
        items = args['latent_vars'].strip('[]').split(',')
        args['latent_vars'] = [s.strip(' \'"') for s in items]
    return args


if __name__ == '__main__':
    import sys, pprint
    args = parse_args(sys.argv[1:])
    pprint.pprint(toydata(**args))
