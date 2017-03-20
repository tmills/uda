#!/bin/bash

perl -pe 's/^\S+\s+//' | perl -pe 's/(\d+):(\S+)/\1/g;s/ /\n/g' | sort -n | uniq -c | sort -n | tail -51 | head -50 | awk '{print $2}' | sort -n
