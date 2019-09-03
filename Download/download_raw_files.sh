#!/bin/bash
###############################################
#Script file with args: train or valid
# ==> File will download and move current 
#     depth folder into image date folder 
#     with raw data. ATTENTION: only raw 
#     data with the same name xxx_sync as 
#     in the depth completion dataset
#     will be downloaded! 
###############################################
###
# file structure should be like this
# Download
#    |--download_raw_files.sh
#

function download_files(){
    mkdir -p '../Data'
    cd '../Data'
    wget 'http://www.cvlibs.net/download.php?file=data_depth_annotated.zip'
    wget 'http://www.cvlibs.net/download.php?file=data_depth_velodyne.zip'
    wget 'http://www.cvlibs.net/download.php?file=data_depth_selection.zip'
}

function unzip_files(){
    cd '../Data'
    unzip 'data_depth_annotated.zip'
    unzip 'data_depth_velodyne.zip'
    unzip 'data_depth_selection.zip'

}


function Download_files(){
	files=($@)
	for i in ${files[@]}; do
		if [ ${i:(-3)} != "zip" ]; then
				date="${i:0:10}"
				name=$(basename $i /)
				shortname=$name'.zip'
				#shortname=$i'.zip'
				fullname=$(basename $i _sync)'/'$name'.zip'
				echo 'shortname: '$shortname
		else
			echo 'Something went wrong. Input array names are probably not correct! Check this manually!'
		fi
		echo "Downloading: "$shortname
		wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
		unzip -o $shortname
		rm -f $shortname
		mv $i'proj_depth' $date'/'$name

        # Remove first 5 and last 5 files of camera images
        cd $date'/'$name'/image_02/data' 
        ls | sort | (head -n 5) | xargs rm -f
        ls | sort | (tail -n 5) | xargs rm -f
        cd '../../image_03/data'
        ls | sort | (head -n 5) | xargs rm -f
        ls | sort | (tail -n 5) | xargs rm -f
        cd ../../../../

		rm -rf $name
	done
}

unzip_files

cd '../Data/train'
train_files=($(ls -d */ | sed 's#/##'))
echo ${files[@]}
Download_files ${train_files[@]}

cd '../val'
valid_files=($(ls -d */ | sed 's#/##'))
Download_files ${valid_files[@]}

