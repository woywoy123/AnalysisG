#!/bin/bash

rm -r *.gch
rm a.out

g++ -Wall -g \
    mtx.h mtx.cxx \
    nunu.h nunu.cxx   \
    base.h base.cxx   \
    particle.h particle.cxx   \
    matrix.h matrix.cxx   \
    tools.h tools.cxx  \
    lm.h lm.cxx \
    main.cxx 

./a.out
#valgrind --leak-check=yes --track-origins=yes --show-leak-kinds=all ./a.out 

#rm -r CMakeFiles
#rm CMakeCache.txt
#rm MakeFile
#rm libcnunu.a
#rm *.so
#rm *.cpp
#rm *.cmake
#
#
#cmake . && make -j12
