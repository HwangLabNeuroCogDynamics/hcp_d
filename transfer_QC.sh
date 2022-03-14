argon_dir=/mnt/nfs/lss/lss_kahwang_hpc/data/HCP_D/
argon_fmriprep=${argon_dir}fmriprep/
mege_dir=/data/backed_up/shared/HCP_D_QualityControl/
mege_fmriprep=${mege_dir}fmriprep/

cd $argon_fmriprep
subjects=($(ls -d sub-*/))
echo "${subjects[@]}"
for subject in "${subjects[@]}"
do
  echo $subject
  mkdir -v ${mege_fmriprep}${subject}
  cp -rv ${argon_fmriprep}${subject}figures/ ${mege_fmriprep}${subject}
done
