for i in {0..99}; do
    python extract_video_mask.py --data_dir tapvid_benchmark/tapvid_kinetics_data_strided.pkl --idx $i
done