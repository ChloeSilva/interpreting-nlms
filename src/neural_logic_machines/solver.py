import neural_logic_machines.architecture as architecture
import neural_logic_machines.file_processor as file_processor
import neural_logic_machines.interpreter as interpreter
import math
import jax.numpy as jnp
from jax import random

class Solver:

    def __init__(self, problem, depth, predictor):
        self.solution = []
        self.training_data = []
        self.problem = problem
        self.depth = depth
        self.predictor = predictor

    # Trains the neural network and saves the final parametes in solution  
    def train(self, training_path, learning_rate=1e-2, batch_size=100, num_epochs=1):
        with open(training_path) as f:
            training_data = [line.strip() for line in f]
    
        processed_data = file_processor.process(training_data)
        io_tensors = [self.problem.text_to_tensor(i, o) 
                      for i, o in processed_data ]
        self.training_data = io_tensors
        
        weights = self.init_weights(random.PRNGKey(0))
        # weights = [[(jnp.zeros(w.shape), jnp.zeros(b.shape)) for w, b in l] for l in weights]

        for epoch in range(num_epochs):
            print(f'starting epoch {epoch}')
            for i, o in self.get_batches(io_tensors, batch_size):
                weights = architecture.update(self.predictor, weights, i, o, learning_rate)
            print(f'layer 1: {weights[0][0]}')
            print(f'layer 1: {weights[0][1]}')
            print(f'layer 1: {weights[0][2]}')
            print(f'epoch: {epoch}')
    
        self.solution = weights
    
    # Runs the network on a single input and returns the output
    def run(self, input_path, threshold):
        with open(input_path) as f:
            input_data = [line.strip() for line in f]

        instance = self.problem.create_instance(input_data)
        input_tensor = instance.text_to_tensor(input_data)
        output = architecture.predict_nlm(self.solution, input_tensor)
        output = [jnp.where(o > threshold, 1, 0) for o in output]

        return instance.tensor_to_text(output)
    
    # Runs the neural network on the test data and returns the accuracy
    def test(self, test_path, threshold):
        with open(test_path) as f:
            test_data = [line.strip() for line in f]

        processed_data = file_processor.process(test_data)
        io_tensors = [self.problem.text_to_tensor(i, o) 
                      for i, o in processed_data]
        
        correct = 0
        for i, expected_o in  io_tensors:
            o = self.predictor(self.solution, i)
            o = [jnp.where(o > threshold, 1.0, 0.0) for o in o]
            if all([jnp.array_equal(a, b) for a, b in zip(o, expected_o)]):
                correct += 1

        return correct/len(io_tensors)
        
    
    def get_batches(self, data, size):
        # list(zip(*[a,b,c])) ==> list(zip(a,b,c)) => [(a1,b1,c1), (a2,b2,c2), etc]
        return [list(zip(*(data[i:i + size]))) for i in range(0, len(data), size)]

    
    def init_neural_unit(self, num_preds, arity, key):
        w_key, b_key = random.split(key)
        return (random.uniform(w_key, (num_preds, num_preds*math.factorial(arity))),
                random.uniform(b_key, (num_preds,)))
    
    def init_layer(self, key):
        num_units = len(self.problem.max_internal)
        keys = random.split(key, num_units)
        return [self.init_neural_unit(m, n, k) for m, n, k in 
                zip(self.problem.max_internal, range(num_units), keys)]

    def init_weights(self, key):
        keys = random.split(key, self.depth)
        return [self.init_layer(k) for k in keys]
 
                 
class NLM(Solver):
    
    def __init__(self, problem, depth):
        super().__init__(problem, depth, architecture.predict_nlm)

    def interpret(self, threshold):
        i = interpreter.Interpreter(self.problem, 
                                    threshold, 
                                    self.depth, 
                                    interpreter.Interpreter.interpret_normal_rules,
                                    self.training_data)
        return i.interpret_params(self.solution)

class MNLM(Solver):
    
    def __init__(self, problem, depth):
        super().__init__(problem, depth, architecture.predict_mnlm)
    
    def interpret(self, threshold):
        i = interpreter.Interpreter(self.problem, 
                                    threshold, 
                                    self.depth, 
                                    interpreter.Interpreter.interpret_definite_rules,
                                    self.training_data)
        return i.interpret_params(self.solution)