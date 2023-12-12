# Define pckg name
PACKAGE_NAME = "xclip"

all:
	gcc -o xclip xclip.c -lX11 -lXmu -DPACKAGE_NAME=$(PACKAGE_NAME)

install: all
	mkdir -p ~/local/bin
	cp xclip ~/local/bin/


