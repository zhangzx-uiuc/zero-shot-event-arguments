# Zero-shot Event Argument Extraction
This repo contains the source code for training and testing for the event argument extraction (EAE) task in a zero-shot manner. In general, we built a shared embedding space for both mentions in a sentence and type names in the ontology. Our system only requires the event argument names like `Attacker`, `Agent` and `Victim` to conduct zero-shot training and testing on argument role types in all scenarios.

## Usage
`python train.py --name RUN_NAME --train_types ace_train_10 --test_types ace_test_23 --gpu 0`
