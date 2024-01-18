#!/usr/bin/env bash
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH -J Attack_freq
#SBATCH --array=0-36%10
#SBATCH --output=slurm/flc/attack/%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de


echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


CONFIG="Motion_Deblurring/Options/Deblurring_Restormer.yml"

cd Motion_Deblurring
##### NNOOOWWW     NNOOOWWWWWWW

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 3 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 5 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 10 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 20 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth

elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 3 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 5 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 10 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 20 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth




elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 10 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 3 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 11 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 5 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 12 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 10 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 13 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 20 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 14 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 3 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 15 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 5 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 16 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 10 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 17 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 20 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_False_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth







elif [[ $SLURM_ARRAY_TASK_ID -eq 18 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 19 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 3 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 20 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 5 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 21 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 10 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 22 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 20 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth

elif [[ $SLURM_ARRAY_TASK_ID -eq 23 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 3 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 24 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 5 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 25 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 10 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 26 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 20 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_FreqAvgUp/models/net_g_latest.pth



elif [[ $SLURM_ARRAY_TASK_ID -eq 27 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha  --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 28 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 3 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 29 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 5 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 30 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 10 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 31 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method pgd --iterations 20 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth



elif [[ $SLURM_ARRAY_TASK_ID -eq 32 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 3 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 33 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 5 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
elif [[ $SLURM_ARRAY_TASK_ID -eq 34 ]]
then
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 10 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
else
    python -W ignore test.py --half_precision --test_wo_drop_alpha --attack --method cospgd --iterations 20 --gpu_id 0 --result_folder helix_corrected_upsampling --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp' --weights /gpfs/bwfor/work/ws/ma_sagnihot-projects/freq-restormer/experiments/Deblurring_Restormer_flc_pooling_low_freq_True_concat_alpha_learned_with_blur_without_drop_alpha_Adversarial_training_True_pixel_shuffle_zero_padding_upsampling_SplitUp/models/net_g_latest.pth
fi


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
