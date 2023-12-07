import jax.numpy as jnp
import neural_logic_machines.architecture as architecture
from collections import namedtuple
from itertools import combinations, permutations
import math

Rule = namedtuple('Rule', ['head', 'body'])

class Uninterpretable(Exception):
    pass

def logit(x: float):
    """Returns the inverse sigmoid"""
    return math.log(x/(1-x))

class Interpreter:
    def __init__(self, problem, threshold, depth, interpret_func, data):
        self.problem = problem
        self.lower = logit(threshold) ** depth
        self.upper = logit(threshold) ** (1/depth)
        self.interpret_func = interpret_func
        self.data = data
        self.current_solution = None

    def interpret_params(self, solution) -> str:
        self.current_solution = solution
        try: 
            rules = [self.interpret_layer(layer) for layer in solution]
        except Uninterpretable as e:
            return f'Uninterpretable parameters: {e}'
        
        dependencies = self.check_dependencies(rules)
        if dependencies:
            return self.rules_to_text_with(dependencies, rules)

        return self.rules_to_text(rules)

    def interpret_layer(self, layer) -> list[list[Rule]]:
        rules = [self.interpret_unit(i, w, b) for i, (w, b) in enumerate(layer)]
        return rules

    def interpret_unit(self,
                       arity: int,
                       weights: jnp.ndarray, 
                       biases: jnp.ndarray) -> list[Rule]:
        rules = [self.interpret_func(self, arity, h, w, b) 
                for h, (w, b) in enumerate(zip(weights, biases))]
        return sum(rules, [])
    
    def interpret_definite_rules(self,
                                 arity: int,
                                 head: int, 
                                 weights: jnp.ndarray, 
                                 bias: int) -> list[Rule]:
        weights = list(map(abs, weights))
        indexed_weights = reversed(sorted(enumerate(weights), key=lambda x: x[1]))

        bodies = []
        current = [[]]
        for w in indexed_weights:
            current = [c + [w] for c in current]
            i = 0
            length = len(current)
            while i < length:
                body = current[i]
                if sum([x[1] for x in body]) + bias > self.upper:
                    if self.real_body(head, list(zip(*body))[0], arity):
                        bodies.append([x[0] for x in body])
                        del body[-1]
                    else:
                        del current[i]
                        current += list(map(list, combinations(body, len(body)-1)))
                i += 1

        in_rules = set([i for body in bodies for i in body])
        not_in_rules = set(range(len(weights))) - in_rules

        combs = sum([list(combinations(in_rules, n))
                     for n in range(1, len(in_rules))], [])
        
        base = sum([weights[i] for i in not_in_rules]) + bias
        for comb in combs: 
            comb = list(comb)
            if sum([weights[i] for i in comb]) + base > self.lower:
                body = comb + list(not_in_rules)
                if self.real_body(head, body, arity):
                    raise Uninterpretable("Ambiguous probabilites.")

        return [Rule(head, body) for body in bodies]
    
    def interpret_normal_rules(self, 
                               arity: int,
                               head: int, 
                               weights: jnp.ndarray, 
                               bias: int) -> list[Rule]:
        pass

    def check_dependencies(self, rules):
        return []

    def rules_to_text(self, rules: list[list[Rule]]) -> str:
        text_rules = []
        max_pred = self.problem.max_predicates
        names = self.get_names(self.problem.predicate_names)
        for layer in rules:
            prev = 0
            for i, (name, unit) in enumerate(zip(names, layer)):
                expanded = prev
                prev = max_pred[i]
                perms = list(permutations(range(i)))
                for head, body in unit:
                    rule = name[head] + self.perm_to_text(perms[0]) + ' :- '
                    for atom in body:
                        rule += name[atom//len(perms)]
                        rule += self.perm_to_text(perms[atom%len(perms)])
                        if atom//len(perms) < expanded:
                            rule = rule[:-5] + ')'
                        rule += ', '
                    text_rules.append(rule[:-2] + '.')
        return text_rules


    def rules_to_text_with(self, dependencies, rules):
        pass

    def get_names(self, names: list[list[str]]) -> list[list[str]]:
        new = [[], []] + names, [[]] + names + [[]], names + [[], []]
        return [sum(n, []) for n in zip(*new)][1:-1]
    
    def perm_to_text(self, perm) -> str:
        if perm == ():
            return ''
        return '('+''.join(['X'+str(i)+', ' for i in perm])[:-2] + ')'
    
    def real_body(self, head, body, arity) -> bool:
        # TODO: this is only relevant for unit tests
        if self.data == []:
            return True
        
        network = [(jnp.zeros_like(w), jnp.zeros_like(b)) 
                   for w, b in self.current_solution[0]]
        
        for i in body:
            network[arity] = (network[arity][0].at[(head, i)].set(1), 
                              network[arity][1].at[head].set(-(len(body)-1)*1))

        facts = []
        data = sum(list(zip(*self.data)), ())
        
        for input in data:
            output = architecture.predict_nlm([network], input)
            facts += output

        for output in facts:
            if (output > 0.5).any():
                return True
            
        return False
        