import neural_logic_machines.interpreter as interpreter

class Solution:
    def __init__(self,
                 accuracy: float,
                 inputs: list[Facts],
                 outputs: list[Facts]):
        self.accuracy = accuracy
        self.inputs = interpreter.get_facts(inputs)
        self.outputs = interpreter.get_facts(outputs)

    def __str__(self):
        pass