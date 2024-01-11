#!/usr/bin/env bash
#SBATCH --time=119:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH -J Train_freq
#SBATCH --array=0-3%4
#SBATCH --output=slurm/flc/train/%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de


echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


CONFIG="Motion_Deblurring/Options/Deblurring_Restormer.yml"


##### NNOOOWWW     NNOOOWWWWWWW

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then
    python -W ignore basicsr/train.py -opt $CONFIG --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp'

elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    python -W ignore basicsr/train.py -opt $CONFIG --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'FreqAvgUp'

elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    python -W ignore basicsr/train.py -opt $CONFIG --flc --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp'

else
    python -W ignore basicsr/train.py -opt $CONFIG --flc --adversarial --use_alpha --learn_alpha --blur --gpu_id 0 --zero_padding --upsample_method 'SplitUp'
fi


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime