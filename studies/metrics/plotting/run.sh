#!/bin/bash

for ((i=0; i<8; i++))
do
    for ((k=0; k<201; ++k)) 
    do
        for ((t=0; t<10; ++t))
        do
            python run.py $k $t $i 
        done
    done
done

