# Makefile for building documentation
.PHONY: html doxygen

html:
	sphinx-build -b html docs docs/_build/html

doxygen:
	doxygen Doxyfile

all: doxygen html