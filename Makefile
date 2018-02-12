
space :=
space +=

source = $(firstword $(subst +, ,$1))
target = $(word 2, $(subst +, ,$1))
pivotmethod = $(word 3, $(subst +, ,$1))
sourcetarget = $(subst $(space),+,$(wordlist 1, 2, $(subst +, ,$1)))
joindot = $(subst $(space),.,$(join $1,$2))

.PRECIOUS: %_reduced.liblinear0
# for a target like seed+strat_50k/training-data_reduced.liblinear0, the
# prereq is seed+strat/training-data.liblinaer
.SECONDEXPANSION:
%/training-data_reduced.liblinear0: $$(firstword $$(subst _, ,%))/training-data.liblinear
	mkdir -p $*
	python scripts/reindex_liblinear.py $^ $@

## Gt50 pivot selection method is symmetrical -- doesn't change based on which
## domain is source and which is target
.PRECIOUS: %/gt50feats.pivots %/random.pivots %/mi-forward.pivots %/mi-backward.pivots
%/gt50feats.pivots: %/training-data_reduced.liblinear0
	python scripts/create_freq_pivots.py $^ > $@

%/gt50feats-forward.pivots: %/gt50feats.pivots
	cp $^ $@

%/gt50feats-backward.pivots: %/gt50feats.pivots
	cp $^ $@

## mi pivot selection is assymetric -- can't use MI against label for target.
## so we need to allow for both directions.
%/mi-forward.pivots: %/training-data_reduced.liblinear0
	python scripts/create_mi_pivots.py $^ 0 > $@

%/mi-backward.pivots: %/training-data_reduced.liblinear0
	python scripts/create_mi_pivots.py $^ 1 > $@

%/random.pivots: %/training-data_reduced.liblinear0
	python scripts/create_random_pivots.py $^ > $@

%/random-forward.pivots: %/random.pivots
	cp $^ $@

## Have it depend on forward instead of plain because make may delete the intermediate file
%/random-backward.pivots: %/random-forward.pivots
	cp $^ $@

## Pivots based on features that are not token-specific:
%/non_token-forward.pivots: %/training-data_reduced.liblinear0
	grep -v "Bag" $*/reduced-features-lookup.txt | grep -v "Following" | grep -v "Preceding" | grep -v "Covered" | grep -v "TreeFrag" | grep -v "Domain" | grep -v "SCORE" | perl -pe 's/^[^:]+ : (\d+)/\1/g' > $@

%/non_token-backward.pivots: %/non_token-forward.pivots
	cp $^ $@

%/non_terminal-forward.pivots: %/training-data_reduced.liblinear0
	grep "TreeFrag" $*/reduced-features-lookup.txt | grep -v "[a-z])" | perl -pe 's/^.+ : (\d+)/\1/g' > $@

%/non_terminal-backward.pivots: %/non_terminal-forward.pivots
	cp $^ $@

stopword_patterns.txt: stopwords.txt
	cat $^ | perl -pe 's/^(.*)$$/_\1 /g' > $@

%/stopword-forward.pivots: stopword_patterns.txt %/training-data_reduced.liblinear0 
	grep -f $< $*/reduced-features-lookup.txt | perl -pe 's/.*: (\d+)/\1/g' > $@

%/stopword-backward.pivots: %/stopword-forward.pivots
	cp $^ $@

.PRECIOUS: %_pivots_done.txt
%_pivots_done.txt: %.pivots $$(dir %.pivots)training-data_reduced.liblinear0
	mkdir -p $*_pivot_data
	python scripts/build_pivot_training_data.py $^ $*_pivot_data > $@

.PRECIOUS: %_theta_full.pkl
%_theta_full.pkl: %_pivots_done.txt
	python scripts/learn_scl_weights.py $*_pivot_data $@

.PRECIOUS: %_theta_svd.pkl
%_theta_svd.pkl: %_theta_full.pkl
	python scripts/reduce_matrix.py $^ $@

%-forward.scl.eval:  $$(dir %-forward.pivots)training-data_reduced.liblinear0 %-forward.pivots %-forward_theta_svd.pkl
	python scripts/eval_scl.py $^ > $@

%-backward.scl.eval: $$(dir %-backward.pivots)training-data_reduced.liblinear0 %-backward.pivots %-backward_theta_svd.pkl
	python scripts/eval_scl.py $^ True > $@

#%.joint.eval: %
%.scl.eval: %-forward.scl.eval %-backward.scl.eval #$$(dir %.pivots)training-data_reduced.liblinear0 %.pivots %_theta_svd.pkl
	cat $^ > $@

#%.eval: %.pivots $$(call source,%).liblinear0 $$(call target,%).liblinear0 pivot_data/%/theta_svd.pkl
#	python scripts/transform_features.py $(call source,$*).liblinear0 $(call target,$*).liblinear0 pivot_data/$*/ $*.pivots > $@

%.compare: $$(call source,%).liblinear0 $$(call target,%).liblinear0 %.pivots
	python scripts/compute_adaptation_variables.py $^ > $@
