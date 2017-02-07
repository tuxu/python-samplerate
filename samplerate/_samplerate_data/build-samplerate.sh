#!/bin/sh
export PATH=$(pwd)/"usr/bin:$PATH"
for TARGET in x86_64-w64-mingw32.static i686-w64-mingw32.static
do
    $TARGET-gcc -O2 -shared -o libsamplerate-$TARGET.dll -Wl,--whole-archive -lsamplerate -Wl,--no-whole-archive
    $TARGET-strip libsamplerate-$TARGET.dll
    chmod -x libsamplerate-$TARGET.dll
done
mv libsamplerate-x86_64-w64-mingw32.static.dll libsamplerate-64bit.dll
mv libsamplerate-i686-w64-mingw32.static.dll libsamplerate-32bit.dll
