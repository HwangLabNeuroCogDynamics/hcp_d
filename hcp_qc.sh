subject=$1
data_dir='/data/backed_up/shared/HCP_D_QualityControl/'
fmriprep_dir=${data_dir}fmriprep/
mriqc_dir=${data_dir}mriqc/

cd $mriqc_dir
google-chrome ${mriqc_dir}sub-${subject}*.html
google-chrome ${fmriprep_dir}sub-${subject}.html
