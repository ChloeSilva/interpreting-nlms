import unittest
import jax.numpy as jnp
import neural_logic_machines.architecture as architecture

class TestArchitecture(unittest.TestCase):

    def test_permute_unary(self):
        # Given
        predicates = jnp.array([[1, 0, 0, 0]])

        # When
        result = architecture.permute_predicate(predicates)

        # Then
        self.assertTrue((result == jnp.array([[1, 0, 0, 0]])).all())

    def test_permute_binary(self):
        # Given
        predicates = jnp.array([[[0, 0, 1, 0],
                           [0, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]]])
        
        # When
        result = architecture.permute_predicate(predicates)

        # Then
        self.assertTrue(
            (result ==
             jnp.array([[[0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]],
                       [[0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]]])).all())
        
    def test_expand(self):
        # Given
        predicates = jnp.array([[0, 1, 0, 1], [0, 0, 1, 1]])

        # When
        result = architecture.expand(predicates)

        # Then
        self.assertTrue(
            (result ==
             jnp.array([[[0, 1, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 0, 1]],
                       [[0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1]]])).all())
        
    def test_reduce(self):
        # Given
        predicates = jnp.array([[[1, 0, 1, 0],
                                [0, 0, 1, 1],
                                [0, 0, 1, 0],
                                [0, 0, 1, 1]],
                               [[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0]]])

        # When
        result = architecture.reduce(predicates)

        # Then
        self.assertTrue((result ==
             jnp.array([[1, 0, 1, 1], 
                        [0, 1, 0, 0]])).all())
        
    def test_predict_1(self):
        # Given
        # sibling(X, Y) :- brother(X, Y).
        # sibling(X, Y) :- sister(X, Y).
        weights = [[(jnp.array([]), jnp.array([])), (jnp.array([]), jnp.array([])), (jnp.array([[0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [1, 0, 0, 0, 0, 0]]),
                                          jnp.array([0, 0, -0.5]))],
                   [(jnp.array([]), jnp.array([])), (jnp.array([]), jnp.array([])), (jnp.array([[0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 1, 0, 0, 0]]),
                                          jnp.array([0, 0, -0.5]))]]
        
        facts = [jnp.empty((0)),
                 jnp.empty((0, 4)),
                 jnp.array([[[0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]])]

        # When
        result = architecture.predict_mnlm(weights, facts)

        # Then
        self.assertTrue(result[0].size == 0)
        self.assertTrue(result[1].size == 0)
        self.assertTrue((jnp.round(result[2]) == jnp.array([[[0, 1, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 0, 0]],
                                                 [[0, 0, 0, 0],
                                                  [1, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 1, 0]],
                                                 [[0, 1, 0, 0],
                                                  [1, 0, 0, 0],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 1, 0]]])).all())

    def test_predict_2(self):
        # Given
        # layer 1: sibling(X, Y) :- sibling(Y, X)
        # layer 2: brother(X, Y) :- male(X), sibling(X, Y)
        weights = [[(jnp.array([]), jnp.array([])), 
                    (jnp.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]), 
                     jnp.array([0, 0, 0, 0, 0])), 
                    (jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # male(X)
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # female(X)
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # brother(X,Y)
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sister(X,Y)
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]), # sibling(X,Y)
                     jnp.array([0, 0, 0, 0, 0]))],
                   [(jnp.array([]), jnp.array([])), 
                    (jnp.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]), 
                     jnp.array([0, 0, 0, 0, 0])), 
                    (jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # male(X)
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # female(X)
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1], # brother(X,Y)
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sister(X,Y)
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), # sibling(X,Y)
                     jnp.array([0, 0, -1.5, 0, 0]))]]
        
        # female(alice).
        # male(bob).
        # female(carol).
        # male(dave).
        # sibling(alice, bob).
        # sibling(carol, dave).
        facts = [jnp.empty((0)),
                 jnp.array([[0, 1, 0, 1],
                            [1, 0, 1, 0]]),
                 jnp.array([[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0]]])]

        # When
        result = architecture.predict_mnlm(weights, facts)

        # Then
        self.assertTrue(result[0].size == 0)
        self.assertTrue((jnp.round(result[1]) == jnp.array([[0, 1, 0, 1],
                                                 [1, 0, 1, 0]])).all())
        self.assertTrue((jnp.round(result[2]) == jnp.array([[[0, 1, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 0, 0]],
                                                 [[0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]],
                                                 [[0, 1, 0, 0],
                                                  [1, 0, 0, 0],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 1, 0]]])).all())

    def test_update(self):
        # Given
        weights = [[(jnp.array([]), jnp.array([])), (jnp.array([]), jnp.array([])), (jnp.array([[0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]]).astype(jnp.float32),
                                          jnp.array([0, 0, 0]).astype(jnp.float32))],
                   [(jnp.array([]), jnp.array([])), (jnp.array([]), jnp.array([])), (jnp.array([[0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]]).astype(jnp.float32),
                                          jnp.array([0, 0, 0]).astype(jnp.float32))]]
        
        facts = [[jnp.empty((0)),
                 jnp.empty((0, 4)),
                 jnp.array([[[0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]]).astype(jnp.float32)]]
        
        target_facts = [[jnp.empty((0)),
                        jnp.empty((0, 4)),
                        jnp.array([[[0, 1, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 0]],
                                   [[0, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 1, 0]],
                                   [[0, 1, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0]]]).astype(jnp.float32)]]

        learning_rate = 100
        
        # When
        for _ in range(10000):
            weights = architecture.update(architecture.predict_mnlm, weights, facts, target_facts, learning_rate)

        # Then
        self.assertTrue((target_facts[0][2] == jnp.round(architecture.predict_mnlm(weights, facts[0])[2])).all())
        # print(f'{architecture.predict_nlm(weights, facts[0])[2]}')

if __name__ == '__main__':
    unittest.main()