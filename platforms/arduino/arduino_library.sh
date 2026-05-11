#!/bin/sh

rm -f src/*.c
rm -f src/*.h

cp -r ../../src src/ 2>/dev/null
cp -r ../../examples examples/ 2>/dev/null

cp aifes_config.cpp src/ 2>/dev/null
cp aifes_config.h src/ 2>/dev/null

echo "Done."