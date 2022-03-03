from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'conv',
    'conv_2x',
    'dwconv',
    'dwsblock',
    'fusedblock',
]
