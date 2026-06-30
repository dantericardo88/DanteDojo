"""dojo/domains — vertical domain layers that plug into the DanteDojo training/eval engine.

Each domain provides a formal rule oracle, a constraint solver, a synthetic data
generator, and a verifier. The verifier is written once and serves three roles:
synthetic-data labeler, RL reward signal, and eval judge.
"""
