"""LangSmith evaluation harness for the deep research agent.

`pytest` covers the wiring and the CLI's branching logic; it deliberately does
not grade the agent's *output*. This package is the other half: it runs the real
agent against a dataset of research questions and scores both what it did (the
tool trajectory) and what it said (citations, responsiveness).

    uv run python -m evals --upload    # create/sync the dataset in LangSmith
    uv run python -m evals --run       # run the agent over it and score

Nothing here is imported by the app.
"""
