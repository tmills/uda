

source = $(firstword $(subst +, ,$1))
target = $(word 2, $(subst +, ,$1))

.SECONDEXPANSION:

%.eval: $$(call source,$$*).liblinear $$(call target,$$*).liblinear 
	python scripts/evaluate_scl.py $^ pivot_data/$*/ > $@

