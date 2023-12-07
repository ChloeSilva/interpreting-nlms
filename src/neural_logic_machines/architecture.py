import jax.numpy as jnp
from jax import grad, jit, vmap
from itertools import permutations
from functools import partial

# This is working with an interleaved representation

def generate_permutations(n: int) -> list[tuple[int, ...]]:
    '''Generates list of all permutations of integers from 0 to n'''
    return list(permutations(range(n)))


def permute_predicate(preds: jnp.ndarray) -> jnp.ndarray:
    '''Generates all permutations of variables for a tensor of predicates'''
    perm = generate_permutations(preds.ndim - 1)
    return jnp.array(sum([[jnp.transpose(pred, p) for p in perm] 
                            for pred in preds], []))

def expand(preds: jnp.ndarray) -> jnp.ndarray:
    '''Applies the expand operator'''
    objects = preds.shape[-1]
    tile_shape = (1, ) * preds.ndim + (objects, )
    final_shape = preds.shape + (objects, )
    return jnp.reshape(jnp.tile(preds, tile_shape), final_shape)

def reduce(preds: jnp.ndarray) -> jnp.ndarray:
    '''Applies the reduce operator'''
    return preds.max(1)

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    '''Sigmoid activation function'''
    return 1/(1+jnp.exp(-x))

def expand_and_reduce(facts: list[jnp.ndarray]) -> list[jnp.ndarray]:
    new_facts = []
    for arity in range(len(facts)):
        temp = facts[arity]
        if arity != 0 and len(facts[arity-1]) != 0:
            temp = jnp.concatenate((expand(facts[arity-1]), temp))

        if arity != len(facts)-1 and len(facts[arity+1]) != 0:
            temp = jnp.concatenate((temp, reduce(facts[arity+1])))
        
        new_facts.append(temp)

    return new_facts

def predict_mnlm(weights: list[jnp.ndarray], facts: list[jnp.ndarray]) -> list[jnp.ndarray]:
    '''forward pass through monotonic nlm'''
    num_pred = [len(fact) for fact in facts]
    big_facts = facts
    for layer in weights:
        big_facts = expand_and_reduce(big_facts)
        new_facts = []
        for arity in range(len(layer)):
            if len(layer[arity][0]) == 0:
                new_facts.append(facts[arity])
                continue
            w, b = layer[arity]
            perm = permute_predicate(big_facts[arity])
            applied = jnp.transpose(jnp.dot(jnp.transpose(perm), jnp.absolute(jnp.transpose(w))))
            outputs = [jnp.maximum(sigmoid(p+b), i) for p, b, i in zip(applied, b, big_facts[arity])]
            if arity > 0:
                outputs = outputs[num_pred[arity-1]:]
            if arity < len(layer) - 1:
                outputs = outputs[:-num_pred[arity+1]]
            new_facts.append(jnp.array(sum([outputs], [])))
        big_facts = new_facts

    return [jnp.maximum(0, big_fact) for big_fact in big_facts]

def predict_nlm(weights: list[jnp.ndarray], facts: list[jnp.ndarray]) -> list[jnp.ndarray]:
    '''forward pass through original nlm'''
    num_pred = [len(fact) for fact in facts]
    big_facts = facts
    for layer in weights:
        big_facts = expand_and_reduce(big_facts)
        new_facts = []
        for arity in range(len(layer)):
            if len(layer[arity][0]) == 0:
                new_facts.append(facts[arity])
                continue
            w, b = layer[arity]
            perm = permute_predicate(big_facts[arity])
            ts = tuple(range(1, arity)) + (0, arity)
            if arity == 0: ts = (0, )
            applied = jnp.dot(w,  jnp.transpose(perm, ts))
            outputs = [sigmoid(p+b) for p, b, i in zip(applied, b, big_facts[arity])]
            if arity > 0:
                outputs = outputs[num_pred[arity-1]:]
            if arity < len(layer) - 1:
                outputs = outputs[:-num_pred[arity+1]]
            new_facts.append(jnp.array(sum([outputs], [])))
        big_facts = new_facts

    return [jnp.maximum(0, big_fact) for big_fact in big_facts]

def loss(predictor):
    '''loss function'''
    def differentiable(weights: list[jnp.ndarray], x, y):
        # prediction = vmap(predictor, in_axes=(None, 0))(weights, x)
        prediction = [predictor(weights, example) for example in x]
        temp = [[-jnp.mean((pa-ya)*(pa-ya)) for pa, ya in zip(p, y)]
                for p, y in zip(prediction, y)]
        return jnp.mean(jnp.array([sum(t) / len(t) for t in temp]))
    return differentiable

def new_weights(learning_rate, w, b, dw, db):
    '''returns updated parameters'''
    if len(w) == 0:
        return (w, b)
    return (w + learning_rate * dw, b + learning_rate * db)

@partial(jit, static_argnums=0)
def update(predictor, weights, x, y, learning_rate):
    '''updates parameters neural network parameters'''
    grads = grad(loss(predictor))(weights, x, y)
    return [[new_weights(learning_rate, w, b, dw, db) for ((w, b), (dw, db)) 
                in zip(lw, lg)] for lw, lg in zip(weights, grads)]