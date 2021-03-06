###### Experiments in unsupervised domain adaptation, starting with
###### structural correspondence learning

## Pipeline for running experiments:
1) Generate liblinear files for each partition of the data using AssertionEvaluation.
Use the new EncoderReusingDataWriter after the first run so that all the feature
spaces have the same encoding, and write the "source" domain training file first
so that all features possible can be found later.
1b) Re-index all these files so that they are indexed at 0 to make everything after this cleaner.
(sklearns functions assume index=0 and this way "feature numbers" are the same as array indices in the code)
> python scripts/reindex_liblinear.py source-training.liblinear

2) Cat together source training with target eval liblinear files
> source+target-training.liblinear

3) Pass that file through extract_pivots_by_frequency.sh (or other pivot selection method)
to get 50 most frequent features.
> cat source+target-training.liblinear | ./scripts/extract_pivots_by_frequency_liblinear.sh > source+target.pivots.txt

at this point it is sometimes helpful to do a sanity check that your source and target files have similar pivots if you run them independently. This will assure you that your step 1 outputs are indeed using the same encoder.

4) Build training data files for each pivot feature classification problem with build_pivot_training_data.pl:
> perl scripts/build_pivot_training_data.pl source+target.pivots.txt source-training.liblinear target-eval.liblinear  pivot_data/source+target

Data will be written to the output folder you specify, here pivot_data/source+target.

Before proceeding any further, make sure you have the python requirements
in the requirements.txt file installed. I have done this in a virtual environment
as follows:

* virtualenv env
* source env/bin/activate
* pip install -r requirements.txt

5) Train pivot classifiers with learn_scl_weights.py. This step performs an
SVD so it may require > 8GB RAM for large feature/instance sets.
> python scripts/learn_scl_weights.py pivot_data/source+target

This step will write the feature projection matrix as a pickled file called theta.pkl

6) Write liblinear files in new feature spaces with transform_features.py:
> python scripts/transform_features.py source-training.liblinear theta.pkl source+target.pivots.txt pivot_data/source+target/transformed

Transformed datasets will be written to the directory in the last argument.

7) Evaluate the different configurations with evaluate_scl.py:
> python scripts/evaluate_scl.py source-training.liblinear target-eval.liblinear pivot_data/source+target/
