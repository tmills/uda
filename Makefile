
space :=
space +=

source = $(firstword $(subst +, ,$1))
target = $(word 2, $(subst +, ,$1))
pivotmethod = $(word 3, $(subst +, ,$1))
sourcetarget = $(subst $(space),+,$(wordlist 1, 2, $(subst +, ,$1)))
joindot = $(subst $(space),.,$(join $1,$2))

.SECONDEXPANSION:
.PRECIOUS: %.liblinear0

%.liblinear0: %.liblinear
	python scripts/reindex_liblinear.py $^ > $@

## Learning pivots based on absolute frequency over two corpora:
%+freq.pivots: $$(call source,%).liblinear0 $$(call target,%).liblinear0
	cat $^ | ./scripts/extract_pivots_by_frequency_liblinear.sh > $@

## Selecting pivots based on those that occur with frequency > 50 in both corpora:
%-gt50feats.txt: %.liblinear0
	cat $^ | perl -pe 's/^\S+\s+//' | perl -pe 's/(\d+):(\S+)/\1/g;s/ /\n/g' | sort -n | uniq -c | sort -n | awk '$$1 >= 50' | awk '{print $$2}' | sort -n  > $@

%+scl.pivots: $$(call source,%)-gt50feats.txt $$(call target,%)-gt50feats.txt
	cat $^ | sort | uniq -c | grep " 2 " | awk '{print $$2}' | grep -v "^0" | sort -n > $@

pivot_data/%/pivots_done.txt: %.pivots $$(call source,%).liblinear0 $$(call target,%).liblinear0
	mkdir -p pivot_data/$*
	python scripts/build_pivot_training_data.py $^ pivot_data/$* > $@

pivot_data/%/theta_svd.pkl: pivot_data/%/pivots_done.txt
	python scripts/learn_scl_weights.py pivot_data/$*

pivot_data/%/transformed/new.liblinear: $$(call source,$$*).liblinear0 pivot_data/%/theta_svd.pkl $$*.pivots
	python scripts/transform_features.py $< pivot_data/$*/ $*.pivots

%.eval: $$(call source,%).liblinear0 $$(call target,%).liblinear0 pivot_data/%/transformed/new.liblinear0
	python scripts/evaluate_scl.py $(call source,$*).liblinear0 $(call target,$*).liblinear0 pivot_data/$*/ > $@
