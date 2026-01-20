export PYTHONPATH=$PWD:$PYTHONPATH

python test_face_xl.py \
--pretrained_model_name_or_path='/path_to_model' \
--prompt None \
--validate_image_path='/test_face_file_path' \
--output_dir='/output_path' \
--minor_color_fix_strength 0.0 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 8 \
--process_size 512 \
--upscale=1  \
--sample_times=1 

