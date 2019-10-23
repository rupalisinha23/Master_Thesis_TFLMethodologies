#!/usr/bin/env bash
echo "Making output directory to store appropriate files"

mkdir -p Output_files
cd Output_files

# Run when noise INCLUSION is desired
# --evaluate without-noise is training on all the classes but evaluating only non-noisy classes
CUDA_VISIBLE_DEVICES=2,3 python /home/rusi02/master_thesis/src/experiment_main.py --vecfile /home/rusi02/master_thesis/data/full_length_sentences.vec --dffile /home/rusi02/research_paper/big_medilytics/data/df_cavr_v1.pkl  --dffile_without_noise  /home/rusi02/master_thesis/data/df_cavr_no_noise.pkl --classes /home/rusi02/research_paper/big_medilytics/data/classes_v1.pkl --classes_without_noise /home/rusi02/master_thesis/data/classes_no_noise.pkl --icf /home/rusi02/master_thesis/data/values_icf.pkl --concepts_dump /home/rusi02/research_paper/big_medilytics/data/concepts_dump.pkl --values_dump /home/rusi02/research_paper/big_medilytics/data/values_dump_with_frequency.pkl --relations_dump /home/rusi02/research_paper/big_medilytics/data/relations_dump.pkl --flag STL --plots no --model CNN --trained_with_noise yes --evaluate with-noise --outputlayers concepts --mtl_type A


# # Run for analysis
# echo "Running the analysis"
# python /home/rusi02/master_thesis/src/analysis.py --outputlayers values --plots no --classes_without_noise /home/rusi02/master_thesis/data/classes_no_noise.pkl --flag STL --evaluate with-noise --train with-noise

#combination_100.vec
#full_length_sentences.vec
