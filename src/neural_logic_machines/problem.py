import jax.numpy as jnp
import neural_logic_machines.instance as instance

class Problem:

    def __init__(self, 
                 max_predicates: list[int],
                 predicate_names: list[list[str]], 
                 knowledge_base: list[str]) -> None:
        self.max_predicates = max_predicates
        self.predicate_names = self.generate_names(predicate_names)
        self.knowledge_base = knowledge_base
        self.max_internal = self.get_max_internal(max_predicates)

    def generate_names(self, predicate_names):
        for names in predicate_names:
            if len(names) != len(set(names)):
                raise RuntimeError("""Cannot have duplicate names for 
                predicates of the same arity""")

        counter = 0
        for max, names in zip(self.max_predicates, predicate_names):
            for j in range(max):
                if len(names) <= j:
                    while names.count(f'p{counter}'):
                        counter += 1
                    names.append(f'p{counter}')
                    counter += 1

        return predicate_names

    def get_max_internal(self, max_pred):
        return [a+b+c for a, b, c in zip(max_pred+[0, 0],
                                         [0]+max_pred+[0], 
                                         [0, 0]+max_pred)][1:-1]

    def create_instance(self, facts):
        return instance.Instance(self, facts)

    def text_to_tensor(self, i, o):
        instance = self.create_instance(i)
        return (instance.text_to_tensor(i), instance.text_to_tensor(o))