import xlsindy

import sympy as sp

t = sp.symbols("t")

Symb = xlsindy.catalog_gen.generate_symbolic_matrix(3,t)

print(Symb)