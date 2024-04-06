from collections import deque


class ThoughtDecomposition:
    def decompose(self, thought):
        # 생각을 분해하는 코드 작성
        pass


class ThoughtGenerator:
    def generate(self, decomposed_thoughts):
        # 분해된 생각을 기반으로 새로운 생각을 생성하는 코드 작성
        pass


class StateEvaluator:
    def evaluate(self, thought):
        # 생각의 가치를 평가하는 코드 작성
        pass


class TreeOfThought:
    def __init__(self):
        self.decomposer = ThoughtDecomposition()
        self.generator = ThoughtGenerator()
        self.evaluator = StateEvaluator()

    def search_bfs(self, initial_thought, max_depth):
        queue = deque([(initial_thought, 0)])
        best_thought = None
        best_value = float('-inf')

        while queue:
            thought, depth = queue.popleft()
            value = self.evaluator.evaluate(thought)

            if value > best_value:
                best_thought = thought
                best_value = value

            if depth < max_depth:
                decomposed = self.decomposer.decompose(thought)
                generated = self.generator.generate(decomposed)
                queue.extend([(g, depth + 1) for g in generated])

        return best_thought