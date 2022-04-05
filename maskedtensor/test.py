import torch
from maskedtensor import masked_tensor

torch.manual_seed(1)

shape = (2,3)
t1 = torch.randn(shape) * -1
t2 = torch.arange(6).reshape(shape)
mask = torch.randint(0, 2, shape).bool()

m1 = masked_tensor(t1, mask)
m2 = masked_tensor(t2, mask)
print (m1)
print (m2)

s1 = t1.to_sparse_coo()
s2 = t2.to_sparse_coo()
sparse_mask = mask.to_sparse_coo()
print (mask)
print (sparse_mask)

m3 = masked_tensor(s1, sparse_mask)
m4 = masked_tensor(s2, sparse_mask)

print (m3)
print (m4)

print("abs 1", torch.abs(m1))
x = torch.abs(m3)
print ("abs", x.layout, x)

# # import itert

# UNARY_NAMES = [
#     "abs",
#     "absolute",
#     "acos",
#     "arccos",
#     "acosh",
#     "arccosh",
#     "angle",
#     "asin",
#     "arcsin",
#     "asinh",
#     "arcsinh",
#     "atan",
#     "arctan",
#     "atanh",
#     "arctanh",
#     "bitwise_not",
#     "ceil",
#     "clamp",
#     "clip",
#     "conj_physical",
#     "cos",
#     "cosh",
#     "deg2rad",
#     "digamma",
#     "erf",
#     "erfc",
#     "erfinv",
#     "exp",
#     "exp2",
#     "expm1",
#     "fix",
#     "floor",
#     "frac",
#     "lgamma",
#     "log",
#     "log10",
#     "log1p",
#     "log2",
#     "logit",
#     "i0",
#     "isnan",
#     "nan_to_num",
#     "neg",
#     "negative",
#     "positive",
#     "pow",
#     "rad2deg",
#     "reciprocal",
#     "round",
#     "rsqrt",
#     "sigmoid",
#     "sign",
#     "sgn",
#     "signbit",
#     "sin",
#     "sinc",
#     "sinh",
#     "sqrt",
#     "square",
#     "tan",
#     "tanh",
#     "trunc",
# ]

# # ret = []
# # bad = []
# # for u in UNARY_NAMES:
# #     try:
# #         x = getattr(torch.sparse, u)
# #         ret.append(u)
# #     except:
# #         bad.append(u)
# # print (ret)
# # print (bad)



