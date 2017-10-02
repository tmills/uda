
space :=
space +=

source = $(firstword $(subst +, ,$1))
target = $(word 2, $(subst +, ,$1))
pivotmethod = $(word 3, $(subst +, ,$1))
sourcetarget = $(subst $(space),+,$(wordlist 1, 2, $(subst +, ,$1)))
joindot = $(subst $(space),.,$(join $1,$2))

.SECONDEXPANSION:
.PRECIOUS: %.liblinear0

%_reduced.liblinear0: %.liblinear
	python scripts/reindex_liblinear.py $^ > $@

%/gt50feats.pivots: %/training-data_reduced.liblinear0
	python scripts/create_freq_pivots.py $^ > $@

## TODO: Update to use single data file and feature groups file
#%+mi.pivots: $$(call source,%).liblinear0 $$(call target,%).liblinear0
#	python scripts/create_mi_pivots.py $^ > $@

%_test.txt: %.pivots
	echo $(dir $*.pivots)/training-data.liblinear0 > $@

%_pivots_done.txt: %.pivots $$(dir %.pivots)training-data_reduced.liblinear0
	mkdir -p $*_pivot_data
	python scripts/build_pivot_training_data.py $^ $*_pivot_data > $@

.PRECIOUS: pivot_data/%/theta_full.pkl
%_theta_full.pkl: %_pivots_done.txt
	python scripts/learn_scl_weights.py $*_pivot_data $@

%_theta_svd.pkl: %_theta_full.pkl
	python scripts/reduce_matrix.py $^ $@

#%.joint.eval: %
%.scl.eval: $$(dir %.pivots)training-data_reduced.liblinear0 %.pivots %_theta_svd.pkl
	python scripts/eval_scl.py $^ > $@

%.eval: %.pivots $$(call source,%).liblinear0 $$(call target,%).liblinear0 pivot_data/%/theta_svd.pkl
	python scripts/transform_features.py $(call source,$*).liblinear0 $(call target,$*).liblinear0 pivot_data/$*/ $*.pivots > $@

%.compare: $$(call source,%).liblinear0 $$(call target,%).liblinear0 %.pivots
	python scripts/compute_adaptation_variables.py $^ > $@
