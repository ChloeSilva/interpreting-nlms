import unittest
import neural_logic_machines.interpreter as interpreter_
import neural_logic_machines.problem as problem_
import neural_logic_machines.file_processor as file_processor
import jax.numpy as jnp 

class TestInterpreter(unittest.TestCase):
    
    def test_interpret_simple(self):
        # Given
        problem = problem_.Problem(max_predicates = [0, 2, 3],
                                  predicate_names = [[],
                                                     ['male', 'female'],
                                                     ['parent', 'son', 'daughter']],
                                  knowledge_base = [])

        threshold = 0.97
        depth = 1
        interpreter = interpreter_.Interpreter(problem,
                                              threshold,
                                              depth,
                                              interpreter_.Interpreter.interpret_definite_rules,
                                              [])
        solution = [[(jnp.array([[0.425, 0.826],
                                   [0.903, 0.174]]),
                      jnp.array([0.431, 0.429])),
                     (jnp.array([[ 0.002, -0.000,  0.000,  0.000, -0.000],
                       [ 0.000,  0.396,  0.000, -0.000,  0.051],
                       [ 0.618,  0.134,  0.553,  0.536,  0.184],
                       [ 0.672,  0.339,  0.819,  0.294,  0.709],
                       [ 0.720,  0.438,  0.323,  0.084,  0.357]]),
                       jnp.array([-7.292, -7.295,  0.297,  0.204,  0.773])),
                     (jnp.array([[ 0.392,  0.341,  0.069,  0.307,  0.113,  0.418,  0.242,  0.067, 0.103,  0.317],
                       [ 0.274,  0.005,  0.611,  0.091,  0.192,  0.322,  0.591,  0.962,0.232,  0.888],
                       [-0.000,  0.000, -0.000,  0.000,  0.931, -0.000,  0.767,  0.298,0.459,  0.159],
                       [-8.044,  0.000,  0.000,  0.000, -0.000,  9.568,  0.036,  0.300,-0.000,  0.288],
                       [-0.000, -0.000, -0.000, -0.000,  0.138,  0.000,  0.179,  0.392,0.960, -0.000]]),
                       jnp.array([-7.164,  -7.324, -7.311, -13.987, -7.314]))]]

        # When
        result = interpreter.interpret_params(solution)

        # Then
        self.assertEqual(result, ['son(X0, X1) :- parent(X1, X0), male(X0).'])

    def test_interpret_definite(self):
        # Given
        problem = problem_.Problem(max_predicates = [0, 2, 3],
                                  predicate_names = [[],
                                                     ['male', 'female'],
                                                     ['father', 'mother', 'daughter']],
                                  knowledge_base = [])
        
        with open("src/unit_tests/data/test_interpreter.txt") as f:
            training_data = [line.strip() for line in f]
    
        processed_data = file_processor.process(training_data)
        io_tensors = [problem.text_to_tensor(i, o) 
                      for i, o in processed_data ]

        threshold = 0.97
        depth = 1
        interpreter = interpreter_.Interpreter(problem,
                                              threshold,
                                              depth,
                                              interpreter_.Interpreter.interpret_definite_rules,
                                              io_tensors)
        solution = [[(jnp.array([[0.425, 0.826],
                                 [0.903, 0.174]]),
                      jnp.array([0.431, 0.429])),
                     (jnp.array([[ 0.002, -0.000,  0.000,  0.000, -0.000],
                                   [ 0.000,  0.396,  0.000, -0.000,  0.051],
                                   [ 0.618,  0.134,  0.553,  0.536,  0.184],
                                   [ 0.672,  0.339,  0.819,  0.294,  0.709],
                                   [ 0.720,  0.438,  0.323,  0.084,  0.357]]),
                       jnp.array([-7.292, -7.295,  0.297,  0.204,  0.773])),
                      (jnp.array([[ 0.392,  0.341,  0.069,  0.307,  0.113,  0.418,  0.242,  0.067, 0.103,  0.317], # male
                                    [ 0.274,  0.005,  0.611,  0.091,  0.192,  0.322,  0.591,  0.962,0.232,  0.888], # female
                                    [-0.000,  0.000, -0.000,  0.000,  0.931, -0.000,  0.767,  0.298,0.459,  0.159], # father
                                    [-0.044,  0.000,  0.000,  0.000, -0.000,  0.568,  0.036,  0.300,-0.000,  0.288], # mother
                                    [-0.000, -0.000, -7.000, -0.000,  0.138,  10.000,  0.179,  10.392,0.960, -0.000]]), # daughter
                       jnp.array([-7.164,  -7.324, -7.311, -7.987, -13.314]))]]
    
        # When
        result = interpreter.interpret_params(solution)

        # Then
        self.assertEqual(result, ['daughter(X0, X1) :- mother(X1, X0), female(X0).', 
                                  'daughter(X0, X1) :- father(X1, X0), female(X0).'])

if __name__ == '__main__':
    unittest.main()