from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'mbconv',
    'dw_dowmup_2x',
    'dw',
    'dw2',
    'dw_dowmup_1x',

]
                           
