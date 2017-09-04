#!/bin/bash

corpora=( "seed-training" "strat-training" "mipacq-training" "i2b2-training" )

for source in "${corpora[@]}"
do
  for target in "${corpora[@]}"
  do
    if [ $source != $target ];
    then
      make $source+$target+gt50feats.eval || exit
    fi
  done
done
