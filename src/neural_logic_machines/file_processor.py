from itertools import groupby
        
def process(d: list[str]) -> list[list[list[str]]]:
        d = list(filter(lambda z: z != '', d))
        d = [tuple(y) for x, y in groupby(d, lambda z: z == 'in:') if not x]
        d = [[list(y) for x, y in groupby(p, lambda z: z == 'out:') if not x] for p in d]
        return d