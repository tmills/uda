#!/usr/bin/perl

use strict;

sub write_instances_for_file{
  my ($pivot, $pivots_hash_pointer, $fn, $fh, $max_index) = @_;
  my %pivots = %{$pivots_hash_pointer};

  my $max_other = get_max_feature_index($fn);

  print "Reading input from $fn\n";
  print "Writing instances for pivot $pivot\n";
  open DATA, " < $fn";
  while(<DATA>){
    chomp;
    ## First lookup value of current pivot to use as our label for this
    ## instance:
    my $label=0;  ## Default is we don't see it in the data so it's = 0
    if(m/ $pivot:(\S+)/){
      my $raw_val = $1;
      if($raw_val > 0.5){
        $label = 1;
      }else{
        $label = 0;
      }
    }
    s/^\S+/$label/;
    s/ $pivot:(\S+)//g;

    ## Now replace the feature weight for every other pivot feature
    for my $other_pivot (keys %pivots){
      s/ $other_pivot:(\S+)//g;
    }

    ## Replace indices above the specified max with empty features.
    for(my $i=$max_index+1; $i <= $max_other; $i++){
      s/ $i:(\S+)//g;
    }

    print $fh "$_\n";
  }
  close DATA;
}

sub get_max_feature_index{
    my ($fn) = @_;
    my $max = 0;
    open DATA, " < $fn";
    while(<DATA>){
      my @fields = split / /;
      my $last_field = $fields[-1];
      my @ind_val = split /:/, $last_field;
      my $ind = int($ind_val[0]);
      if($ind > $max){
        $max = $ind;
      }
    }
    return $max;
}

if ($#ARGV != 3){
  print STDERR "Four required arguments: <pivot file> <source data file> <target data file> <output directory>\n";
  exit
}

print "Reading pivots\n";
my %pivots;
open FP, " < $ARGV[0]";
while(<FP>){
  chomp;
  $pivots{$_} = 1;
}

my $max_index = get_max_feature_index($ARGV[1]);
print "Source data set has max index of $max_index\n";
my $target_max = get_max_feature_index($ARGV[2]);
print "Target data has max index of $target_max\n";

for my $pivot (keys %pivots){
  print "Writing training file for $pivot\n";
  open my $pivot_out, " > $ARGV[3]/pivot_$pivot-training.liblinear";

  print "Writing for seed file:\n";
  write_instances_for_file($pivot, \%pivots, $ARGV[1], $pivot_out, $max_index);
  print "Writing for strat file:\n";
  write_instances_for_file($pivot, \%pivots, $ARGV[2], $pivot_out, $max_index);

  close $pivot_out;
}
close FP;
