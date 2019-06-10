/*
 * Copyright (C) 2007,2008   Alex Shulgin
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
#ifndef PNGPP_SOLID_PIXEL_BUFFER_HPP_INCLUDED
#define PNGPP_SOLID_PIXEL_BUFFER_HPP_INCLUDED

#include <cassert>
#include <cstddef>
#include <climits>
#include <stdexcept>
#include <vector>

#include "config.hpp"
#include "packed_pixel.hpp"
#include "gray_pixel.hpp"
#include "index_pixel.hpp"

namespace png
{

    /**
     * \brief Pixel buffer, that stores pixels as continuous memory chunk.
     * solid_pixel_buffer is useful when user whats to open png, do some
     * changes and fetch to buffer to draw (as texture for example).
     */
    template< typename pixel >
    class solid_pixel_buffer
    {
    public:
        typedef pixel_traits< pixel > pixel_traits_t;
        struct row_traits
        {
            typedef pixel* row_access;
            typedef const pixel* row_const_access;

            static byte* get_data(row_access row)
            {
                return reinterpret_cast<byte*>(row);
            }
        };


        /**
         * \brief A row of pixel data.
         */
        typedef typename row_traits::row_access row_access;
        typedef typename row_traits::row_const_access row_const_access;
        typedef row_access row_type;

        /**
         * \brief Constructs an empty 0x0 pixel buffer object.
         */
        solid_pixel_buffer()
            : m_width(0),
              m_height(0),
              m_stride(0)
        {
        }

        /**
         * \brief Constructs an empty pixel buffer object.
         */
        solid_pixel_buffer(uint_32 width, uint_32 height)
            : m_width(0),
              m_height(0),
              m_stride(0)
        {
            resize(width, height);
        }

        uint_32 get_width() const
        {
            return m_width;
        }

        uint_32 get_height() const
        {
            return m_height;
        }

        /**
         * \brief Resizes the pixel buffer.
         *
         * If new width or height is greater than the original,
         * expanded pixels are filled with value of \a pixel().
         */
        void resize(uint_32 width, uint_32 height)
        {
            m_width = width;
            m_height = height;
            m_stride = m_width * bytes_per_pixel;
            m_bytes.resize(height * m_stride);
        }

        /**
         * \brief Returns a reference to the row of image data at
         * specified index.
         *
         * Checks the index before returning a row: an instance of
         * std::out_of_range is thrown if \c index is greater than \c
         * height.
         */
        row_access get_row(size_t index)
        {
            return reinterpret_cast<row_access>(&m_bytes.at(index * m_stride));
        }

        /**
         * \brief Returns a const reference to the row of image data at
         * specified index.
         *
         * The checking version.
         */
        row_const_access get_row(size_t index) const
        {
            return (row_const_access)(&m_bytes.at(index * m_stride));
        }

        /**
         * \brief The non-checking version of get_row() method.
         */
        row_access operator[](size_t index)
        {
            return (row_access)(&m_bytes[index * m_stride]);
        }

        /**
         * \brief The non-checking version of get_row() method.
         */
        row_const_access operator[](size_t index) const
        {
            return (row_const_access)(&m_bytes[index * m_stride]);
        }

        /**
         * \brief Replaces the row at specified index.
         */
        void put_row(size_t index, row_const_access r)
        {
            row_access row = get_row();
            for (uint_32 i = 0; i < m_width; ++i)
                *row++ = *r++;
        }

        /**
         * \brief Returns a pixel at (x,y) position.
         */
        pixel get_pixel(size_t x, size_t y) const
        {
            size_t index = (y * m_width + x) * bytes_per_pixel;
            return *reinterpret_cast< const pixel* >(&m_bytes.at(index));
        }

        /**
         * \brief Replaces a pixel at (x,y) position.
         */
        void set_pixel(size_t x, size_t y, pixel p)
        {
            size_t index = (y * m_width + x) * bytes_per_pixel;
            *reinterpret_cast< pixel* >(&m_bytes.at(index)) = p;
        }

        /**
         * \brief Provides easy constant read access to underlying byte-buffer.
         */
        const std::vector< byte >& get_bytes() const
        {
            return m_bytes;
        }

#ifdef PNGPP_HAS_STD_MOVE
        /**
         * \brief Moves the buffer to client code (c++11 only) .
         */
        std::vector< byte > fetch_bytes()
        {
            m_width = 0;
            m_height = 0;
            m_stride = 0;

            // the buffer is moved outside without copying and leave m_bytes empty.
            return std::move(m_bytes);
        }
#endif

    protected:
        static const size_t bytes_per_pixel = pixel_traits_t::channels *
                pixel_traits_t::bit_depth / CHAR_BIT;

    protected:
        uint_32 m_width;
        uint_32 m_height;
        size_t m_stride;
        std::vector< byte > m_bytes;

#ifdef PNGPP_HAS_STATIC_ASSERT
        static_assert(pixel_traits_t::bit_depth % CHAR_BIT == 0,
            "Bit_depth should consist of integer number of bytes");

        static_assert(sizeof(pixel) * CHAR_BIT ==
            pixel_traits_t::channels * pixel_traits_t::bit_depth,
            "pixel type should contain channels data only");
#endif
    };

    /**
     * \brief solid_pixel_buffer for packed_pixel is not implemented now.
     * Should there be a gap between rows? How to deal with last
     * useless bits in last byte in buffer?
     */
    template< int bits >
    class solid_pixel_buffer< packed_pixel< bits > >;

} // namespace png

#endif // PNGPP_solid_pixel_buffer_HPP_INCLUDED
