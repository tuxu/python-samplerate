#!/bin/sh
JOBS=8
export PATH="$(pwd)/mxe/usr/bin:$PATH"
for TARGET in x86_64-w64-mingw32.static i686-w64-mingw32.static
do
    make -C mxe libsamplerate -j$JOBS JOBS=$JOBS MXE_TARGETS=$TARGET
    $TARGET-gcc -O2 -shared -o libsamplerate-$TARGET.dll -Wl,--whole-archive -lsamplerate -Wl,--no-whole-archive
    $TARGET-strip libsamplerate-$TARGET.dll
    chmod -x libsamplerate-$TARGET.dll
done
mv libsamplerate-x86_64-w64-mingw32.static.dll libsamplerate-64bit.dll
mv libsamplerate-i686-w64-mingw32.static.dll libsamplerate-32bit.dll
