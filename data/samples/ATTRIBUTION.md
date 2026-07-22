# Sample datasets

All three datasets are **synthetic**, generated deterministically by
`backend/scripts/make_samples.py` (numpy seed 42) and modeled on familiar
real-world shapes (NYC-style taxi trips, daily retail sales, daily weather).
Synthetic data keeps the repo license-clean and makes the golden-eval answers
exactly derivable — see `backend/app/evals/derivations.py`.
