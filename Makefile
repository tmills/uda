

source = $(firstword $(subst +, ,$1))
target = $(word 2, $(subst +, ,$1))

.SECONDEXPANSION:

%.pivots: $$(call source,%).liblinear $$(call target,%).liblinear
	python scripts/reindex_liblinear.py $^
	cat $^ | ./scripts/extract_pivots_by_frequency_liblinear.sh > $@

pivot_data/%/pivots_done.txt: %.pivots $$(call source,%).liblinear $$(call target,%).liblinear
	mkdir -p pivot_data/$*
	perl scripts/build_pivot_training_data.pl $^ pivot_data/$* > $@

pivot_data/%/theta_svd.pkl: pivot_data/%/pivots_done.txt
	python scripts/learn_scl_weights.py pivot_data/$*

pivot_data/%/transformed/new.liblinear: $$(call source,$$*).liblinear pivot_data/%/theta_svd.pkl $$*.pivots
	python scripts/transform_features.py $< pivot_data/$*/theta_svd.pkl $*.pivots pivot_data/$*/transformed

%.eval: $$(call source,%).liblinear $$(call target,%).liblinear pivot_data/%/transformed/new.liblinear
	python scripts/evaluate_scl.py $(call source,$*).liblinear $(call target,$*).liblinear pivot_data/$*/ > $@
