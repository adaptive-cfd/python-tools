#!/bin/bash

if [ "$1" == "" ]; then
	fps=20
else
	fps=$1
fi

for file in *.png
do
        imagew=$(identify -format "%w" $file)
        imageh=$(identify -format "%h" $file)
done

echo "Resolution is "$imagew"x"$imageh "=" $(($imageh*$imagew-10))

mencoder mf://*.png -mf w=$imagew:h=$imageh:fps=$fps:type=png -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=$(($imageh*$imagew*10)) -o video.avi
