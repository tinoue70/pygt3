# What is this ?
This project is to create mini library to read/write GTOOL3 format
file and some utilities to show/convert/plot them.

# GTOOL3 format

There is the GTOOL3 tool collection and library, that is not limited
only read/write library but complete tools to manipulate the output of
GCM. This library uses own binary file format. we call this format as
'gt3' for short.

This file format have been used for a few decades by some climate
models developed in Japan.

# Why 'MINI' ?

Unfortunately original GTOOL3 library is not an open source, and
because of several other reasons, I need my own library to read/write
this file format. So I decided to write new library implmenting
minimum functions.

The original GTOOL3 is written in Fortran, but I choose python for my
version.
