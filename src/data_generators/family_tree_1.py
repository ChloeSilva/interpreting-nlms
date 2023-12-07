import random
from data_generators.objects import names

training_examples = 100
test_examples = 20
min_objects = 5
max_objects = 50
gender_probability = 0.7
related_probability = 0.7
parent_probability = 0.8
son_probability = 0.1
daughter_probability = 0.8

def generate_examples(num_examples, path):	
	with open(path, 'w') as f:
		for _ in range(num_examples):
			num_objects = random.randint(min_objects, max_objects)
			in_ = ''
			out = ''
			for i in range(num_objects):
				gender_flag = False
				if random.uniform(0, 1) < gender_probability:
					if i % 2 == 0:
						in_ += f'female({names[i]}).\n'
					else:
						gender_flag = True
						in_ += f'male({names[i]}).\n'

				if i >= num_objects//2:
					continue

				if random.uniform(0, 1) < related_probability:
					child = names[i]
					parent = names[i+num_objects//2]
					if random.uniform(0, 1) < parent_probability:
						in_ += f'parent({parent}, {child}).\n'
						if gender_flag:
							out += f'son({child}, {parent}).\n'
					if i % 2 == 0:
						if random.uniform(0, 1) < daughter_probability:
							in_ += f'daughter({child}, {parent}).\n'
					else:
						if random.uniform(0, 1) < son_probability:
							in_ += f'son({child}, {parent}).\n'

				if random.uniform(0, 1) < related_probability:
					flip = 1 if i % 2 == 0 else -1
					child = names[i]
					parent = names[i+flip+num_objects//2]
					if random.uniform(0, 1) < parent_probability:
						in_ += f'parent({parent}, {child}).\n'
						if gender_flag:
							out += f'son({child}, {parent}).\n'
					if i % 2 == 0:
						if random.uniform(0, 1) < son_probability:
							in_ += f'daughter({child}, {parent}).\n'
					else:
						if random.uniform(0, 1) < daughter_probability:
							in_ += f'son({child}, {parent}).\n'

			f.write('in:\n'+in_+'\nout:\n'+in_+out+'\n')
	
def main():
	generate_examples(training_examples, 'src/data/family_tree/training_1.txt')
	generate_examples(test_examples,'src/data/family_tree/test_1.txt')
		
if __name__ == "__main__":
	main()