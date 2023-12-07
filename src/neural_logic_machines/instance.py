import jax.numpy as jnp
import numpy as np
from itertools import dropwhile, takewhile
    
class Instance:

    def __init__(self, problem, input):
        self.max_predicates = problem.max_predicates
        self.predicate_names = problem.predicate_names
        self.objects = self.get_objects(input)

    def get_objects(self, text: list[str]) -> list[str]:
        objects = []
        for fact in text:
            args = list(dropwhile(lambda x: x != '(', fact))[1:-2]
            for o in ''.join(args).split(', '):
                if o not in objects:
                     objects.append(o)
        
        return objects

    def text_to_tensor(self, text):
        if text == None:
            return None
        
        tensor = [jnp.zeros((max, ) + ((len(self.objects), )*i)) 
                  for i, max in enumerate(self.max_predicates)]
        
        ones = [[[] for _ in range(i+1)] for i in range(len(self.max_predicates))]
        for fact in text:
            args = self.get_arguments(fact)
            head = self.get_head(fact, len(args))
            index = [head]+list(reversed(args))
            ones[len(args)] = [o+[i] for o, i in zip(ones[len(args)], index)]

        # jit and vmap this
        tensor = [t.at[tuple(map(tuple, i))].set(1) for t, i in zip(tensor, ones)]

        return tensor
    
    def tensor_to_text(self, tensor):
        output = []
        for predicates, names in zip(tensor, self.predicate_names):
            for predicate, name in zip(predicates, names):
                facts = jnp.argwhere(predicate == 1)
                for fact in facts:
                    args = [self.objects[arg] for arg in reversed(fact)]
                    output.append(f'{name}({", ".join(args)}).')
        
        return output

    def get_arguments(self, fact: str) -> list[int]:
        args = list(dropwhile(lambda x: x != '(', fact))[1:-2]
        if len(args) != 0:
            args = [self.objects.index(a) for a in ''.join(args).split(', ')]

        return args
    
    def get_head(self, fact, arity) -> tuple[int, int]:
        head = ''.join(list(takewhile(lambda x: x != '(', fact)))
        if head[-1] == '.':
            head = head[:-1]
        for i, name in enumerate(self.predicate_names[arity]):
            if name == head:
                return i

        raise RuntimeError(f"""Problem description does not contain {arity}-ary
                           predicate {head} found in the data.""")
        
