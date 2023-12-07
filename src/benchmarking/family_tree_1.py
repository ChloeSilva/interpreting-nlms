from neural_logic_machines import problem
from neural_logic_machines import solver
import jax.numpy as jnp

problem_1 = problem.Problem(
        max_predicates = [0, 2, 3],
        predicate_names = [[],
                           ['male', 'female'],
                           ['parent', 'son', 'daughter']],
        knowledge_base = [])
    
def main():
    nlm  = solver.MNLM(problem_1, depth=1)
    nlm.train('src/data/family_tree/training_1.txt', learning_rate=100, 
            batch_size=20, num_epochs=10000)
    solution = nlm.test('src/data/family_tree/test_1.txt', threshold=0.9)
    program = nlm.interpret(threshold=0.9)

    print(f'learned program:\n{program}\nwith accuracy: {solution}')
    print(solution)


if __name__ == "__main__":
    jnp.set_printoptions(precision=3, floatmode="fixed", suppress=True)
    main()