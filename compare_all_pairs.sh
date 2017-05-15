#!/bin/bash

corpora=( "seed-training" "strat-training" "mipacq-training" "i2b2-training" )

for source in "${corpora[@]}"
do
  for target in "${corpora[@]}"
  do
    if [ $source != $target ];
    then
      make $source+$target+scl.compare || exit
    fi
  done
done
