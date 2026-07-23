"""AION application — the operating environment for the platform.

A local web app that surfaces every AION capability the moment it exists. Today
that is the tokenization input path; as the platform grows (BPE, datasets,
training, evaluation, inference) each capability becomes a panel here.

Design split, mirroring the platform's core/shell discipline:

    api.py     pure functions: capability -> JSON-able report (unit-tested,
               no HTTP, no I/O)
    server.py  standard-library HTTP plumbing: routes requests to api, serves
               the static frontend
    static/    the zero-build vanilla frontend

Launch:  python -m aion.app
"""
