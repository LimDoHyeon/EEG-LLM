import argparse
from TreeOfThought.src.tot.methods.bfs import solve
# from TreeOfThought.src.tot.tasks.game24 import Game24Task
from TreeOfThought.src.tot.tasks.crosswords import MiniCrosswordsTask

args = argparse.Namespace(backend='gpt-4', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

task = MiniCrosswordsTask()
ys, infos = solve(args, task, 100)
print(ys[0])