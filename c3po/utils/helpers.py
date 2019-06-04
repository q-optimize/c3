def sum_lambdas(t, cs):
    sig = 0
    for c in cs:
        sig += c(t)
    return sig
