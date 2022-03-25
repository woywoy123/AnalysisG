#!bin/bash

tar -cvf Plots.tar Plots
cp Plots.tar /DESY/plotgallery/
cd /DESY/plotgallery/
tar -xvf Plots.tar 
rm -rf content
mv Plots content
rm Plots.tar
cd -
rm Plots.tar

