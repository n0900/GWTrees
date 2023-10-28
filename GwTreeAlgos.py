import numpy as np
import scipy.stats as st
import time as t

rng = np.random.default_rng(seed=12345)


def rnd_number(distribution):
    match distribution.lower():
        case "poisson":
            return rng.poisson(lam=1)
        case "geometric":
            # -1 because this implementation of the geometric distribution starts at 1 instead of 0
            return rng.geometric(p=0.5) - 1
        case "catalan":
            return int(rng.choice(a=[0, 1, 2], p=[0.25, 0.5, 0.25]))
        case "motzkin":
            # takes values between [0,3) as int -> {0,1,2}
            return rng.integers(low=0, high=3)
        case _:
            print(f"Error! Chosen Distribution: {distribution} not found!")
            return 0


def rnd_vector(distribution, length):
    match distribution.lower():
        case "poisson":
            return rng.poisson(lam=1, size=length)
        case "geometric":
            # -1 because this implementation of the geometric distribution starts at 1 instead of 0
            return rng.geometric(p=0.5, size=length) - 1
        case "catalan":
            return rng.choice(a=[0, 1, 2], p=[0.25, 0.5, 0.25], size=length)
        case "motzkin":
            # takes values between [0,3) as int -> {0,1,2}
            return rng.integers(low=0, high=3, size=length)
        case _:
            print(f"Error! Chosen Distribution: {distribution} not found!")
            return np.zeros(1)


def get_pmf(distribution, limit):
    match distribution.lower():
        case "poisson":
            return st.poisson.pmf(k=range(limit), mu=1)
        case "geometric":
            return st.geom.pmf(k=range(limit), p=0.5, loc=-1)
        case "catalan":
            return np.array([0.25, 0.5, 0.25])
        case "motzkin":
            return np.array([1/3, 1/3, 1/3])
        case _:
            print(f"Error! Chosen Distribution: {distribution} not found!")
            return np.zeros(1)


def get_cdf(distribution, limit):
    match distribution.lower():
        case "poisson":
            return st.poisson.cdf(k=range(limit), mu=1)
        case "geometric":
            return st.geom.cdf(k=range(limit), p=0.5, loc=-1)
        case "catalan":
            return np.array([0.25, 0.75, 1])
        case "motzkin":
            return np.array([1/3, 2/3, 1])
        case _:
            print(f"Error! Chosen Distribution: {distribution} not found!")
            return np.zeros(1)


def tree_to_walk(Xi: np.ndarray) -> np.ndarray:
    S = np.zeros(len(Xi)+1, dtype=int)
    for i in range(len(Xi)):
        S[i+1] = S[i] + Xi[i] - 1
    return S


def rotation(Xi: np.ndarray) -> np.ndarray:
    S = np.zeros(len(Xi)+1)
    for i in range(len(Xi)):
        S[i+1] = S[i] + Xi[i] - 1
    pivot_point = S.argmin()
    return np.concatenate((Xi[pivot_point:], Xi[:pivot_point]))


def get_pvalues(distribution, limit):
    pmf = get_pmf(distribution, limit)
    pvalues = np.zeros(len(pmf))
    p = 1
    for i in range(len(pmf)):
        pvalues[i] = pmf[i]/p
        p -= pmf[i]
    return pvalues


def get_depth(S):
    return round(len(S) - sum(x < y for x, y in zip(S[1:], S)), 1)


def dfs_construction(distribution, limit):
    start = t.perf_counter()
    tries = 0

    while True:
        Xi = np.zeros(limit, dtype=int)
        stack = [0]
        counter = 0
        tries += 1
        i = 0
        while stack and counter < limit:
            stack.pop()
            children = rnd_number(distribution)
            stack += [i for i in range(children)]
            counter += children
            Xi[i] = children
            i += 1

        if (not stack) and (counter == limit-1):
            break

    time = (t.perf_counter()-start)*1000000
    S = tree_to_walk(Xi)
    depth = get_depth(S)

    assert len(S) == limit + 1, f"Invalid Random Walk!\nS = {S}"
    assert (S[:-1] >= 0).all(), f"Invalid Random Walk!\nS = {S}"
    return [",".join([str(i) for i in Xi]), depth, time, tries]


def rnd_walk_construction(distribution, limit):
    start = t.perf_counter()
    tries = 0
    Xi = np.zeros(1, dtype=int)

    while sum(Xi) != limit-1:
        Xi = rnd_vector(distribution, limit)
        tries += 1

    Xi = rotation(Xi)

    time = (t.perf_counter()-start)*1000000
    S = tree_to_walk(Xi)
    depth = get_depth(S)

    assert len(S) == limit + 1, f"Invalid Random Walk!\nS = {S}"
    assert (S[:-1] >= 0).all(), f"Invalid Random Walk!\nS = {S}"
    return [",".join([str(i) for i in Xi]), depth, time, tries]


def mlt_construction(pvalues, limit):
    start = t.perf_counter()
    Z = np.zeros(len(pvalues), dtype=int)
    tries = 0
    total = -999
    k = 0

    while total != limit-1:
        tries += 1
        n = limit
        for i, p in enumerate(pvalues):
            Z[i] = rng.binomial(n, p)
            n -= Z[i]
            if n == 0:
                k = i+1
                total = sum(i * Z[i] for i in range(k))
                break

    Xi = np.array([i for i in range(k) for _ in range(Z[i])])
    Xi = rng.permutation(Xi)
    Xi = rotation(Xi)

    time = (t.perf_counter()-start)*1000000
    S = tree_to_walk(Xi)
    depth = get_depth(S)

    assert len(S) == limit + 1, f"Invalid Random Walk!\nS = {S}"
    assert (S[:-1] >= 0).all(), f"Invalid Random Walk!\nS = {S}"
    return [",".join([str(i) for i in Xi]), depth, time, tries]


# Combinatorical Construction - only for cayley trees
def mlt_construction_poisson(limit):
    start = t.perf_counter()
    # Array rnd-variables between 1-n, length n-1
    Z = rng.integers(low=0, high=limit, size=limit-1)
    # Array length n+1 (0..n) to count the variables in Z incl. zero
    Xi = np.zeros(limit, dtype=int)
    tries = 1

    for i in Z:
        Xi[i] += 1

    Xi = rotation(Xi)

    time = (t.perf_counter()-start)*1000000
    S = tree_to_walk(Xi)
    depth = get_depth(S)

    assert len(S) == limit + 1, f"Invalid Random Walk!\nS = {S}"
    assert (S[:-1] >= 0).all(), f"Invalid Random Walk!\nS = {S}"
    return [",".join([str(i) for i in Xi]), depth, time, tries]
