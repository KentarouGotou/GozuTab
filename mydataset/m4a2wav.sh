#!/bin/bash
cd m4a_files
for file in *.m4a
do
echo ${file%.*}
    ffmpeg -i $file ../audio/${file%.*}.wav
done

# for file in *.m4a
# do
# echo ${file%.*}
# 	ffmpeg -i $file ../audio/${file%.*}.wav
# done