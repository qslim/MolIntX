# Command to download dataset:
#   bash script_download_superpixels.sh


DIR=data/superpixels/
cd $DIR


FILE=superpixels.zip
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/y2qwa77a0fxem47/superpixels.zip?dl=1 -o superpixels.zip -J -L -k
	unzip superpixels.zip -d ./
fi
