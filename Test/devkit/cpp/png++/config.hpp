/*
 * Copyright (C) 2008   Alex Shulgin
 *
 * This file is part of png++ the C++ wrapper for libpng.  PNG++ is free
 * software; the exact copying conditions are as follows:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. The name of the author may not be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
 * NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef PNGPP_CONFIG_HPP_INCLUDED
#define PNGPP_CONFIG_HPP_INCLUDED

#include <stdlib.h>

// Endianness test
#if defined(__GLIBC__)

#include <endian.h>

#elif defined(_WIN32)

#define	__LITTLE_ENDIAN	1234
#define	__BIG_ENDIAN	4321
#define __BYTE_ORDER __LITTLE_ENDIAN

#elif defined(__APPLE__)

#include <machine/endian.h>
#include <sys/_endian.h>

#elif defined(__FreeBSD__)

#include <machine/endian.h>
#include <sys/endian.h>

#elif defined(__sun)

#include <sys/isa_defs.h>

#else

#error Byte-order could not be detected.

#endif

// Determine C++11 features
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && defined(__GXX_EXPERIMENTAL_CXX0X__)

#define PNGPP_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
// gcc c++11 support list
// http://gcc.gnu.org/projects/cxx0x.html

// gcc supports static_assert since 4.3
#if (PNGPP_GCC_VERSION >= 40300)
#define PNGPP_HAS_STATIC_ASSERT
#endif

// gcc supports std::move since 4.6
#if (PNGPP_GCC_VERSION >= 40600)
#define PNGPP_HAS_STD_MOVE
#endif

#undef PNGPP_GCC_VERSION

#elif defined(_MSC_VER)

// MS Visual C++ compiler supports static_assert and std::move since VS2010
// http://blogs.msdn.com/b/vcblog/archive/2011/09/12/10209291.aspx
#if (_MSC_VER >= 1600)
#define PNGPP_HAS_STATIC_ASSERT
#define PNGPP_HAS_STD_MOVE
#endif

#endif


#endif // PNGPP_CONFIG_HPP_INCLUDED
