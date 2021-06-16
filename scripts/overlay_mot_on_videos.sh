mot_seq_dir=/home/infres/chagneux/repos/surfnet/data/validation_videos/T1/segments/mot_gt_files/segments-T1
vid_dir=/home/infres/chagneux/repos/surfnet/data/validation_videos/T1/segments/videos
ourput_dir=/home/infres/chagneux/repos/surfnet/data/validation_videos/T1/segments/videos_gt_overlay
cd $mot_seq_dir

for f in *; do 
    echo $f;
    input_video=${vid_dir}/$f.mp4
    input_mot_file=${mot_seq_dir}/$f/gt/gt.txt
    cd /home/infres/chagneux/repos/surfnet
    python src/overlay_tracking_results_on_video.py --input_video ${input_video} --input_mot_file ${input_mot_file} --output_name $f --write True
done