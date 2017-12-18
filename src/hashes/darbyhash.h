#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>
#include <wmmintrin.h>
#include <x86intrin.h>
#include <cstdint>

#define UNALIGNED_OK 1

#define likely(cond) __builtin_expect(!!(cond), 1)
#define unlikely(cond) __builtin_expect(!!(cond), 0)

#define darbyhash_unreachable() __builtin_unreachable()

/* 'magic' primes */
static const uint64_t darbyhash_p0 = 17048867929148541611ull;
static const uint64_t darbyhash_p1 = 9386433910765580089ull;
static const uint64_t darbyhash_p2 = 15343884574428479051ull;
static const uint64_t darbyhash_p3 = 13662985319504319857ull;
static const uint64_t darbyhash_p4 = 11242949449147999147ull;
static const uint64_t darbyhash_p5 = 13862205317416547141ull;
static const uint64_t darbyhash_p6 = 14653293970879851569ull;

/* rotations */
static const unsigned darbyhash_s0 = 41;
static const unsigned darbyhash_s1 = 17;
static const unsigned darbyhash_s2 = 31;

static __inline uint32_t darbyhash_fetch32_le(const void *v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return *(const uint32_t *)v;
#else
  return bswap32(*(const uint32_t *)v);
#endif
}

static __inline uint16_t darbyhash_fetch16_le(const void *v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return *(const uint16_t *)v;
#else
  return bswap16(*(const uint16_t *)v);
#endif
}

static __inline uint64_t darbyhash_fetch64_le(const void *v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return *(const uint64_t *)v;
#else
  return bswap64(*(const uint64_t *)v);
#endif
}


static __inline uint64_t darbyhash_tail64_le(const void *v, size_t tail) {
  const uint8_t *p = (const uint8_t *)v;
  uint64_t r = 0;
  switch (tail & 7) {
#if UNALIGNED_OK && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  /* For most CPUs this code is better when not needed
   * copying for alignment or byte reordering. */
  case 0:
    return darbyhash_fetch64_le(p);
  case 7:
    r = (uint64_t)p[6] << 8;
  case 6:
    r += p[5];
    r <<= 8;
  case 5:
    r += p[4];
    r <<= 32;
  case 4:
    return r + darbyhash_fetch32_le(p);
  case 3:
    r = (uint64_t)p[2] << 16;
  case 2:
    return r + darbyhash_fetch16_le(p);
  case 1:
    return p[0];
#else
  /* For most CPUs this code is better than a
   * copying for alignment and/or byte reordering. */
  case 0:
    r = p[7] << 8;
  case 7:
    r += p[6];
    r <<= 8;
  case 6:
    r += p[5];
    r <<= 8;
  case 5:
    r += p[4];
    r <<= 8;
  case 4:
    r += p[3];
    r <<= 8;
  case 3:
    r += p[2];
    r <<= 8;
  case 2:
    r += p[1];
    r <<= 8;
  case 1:
    return r + p[0];
#endif
  }
  darbyhash_unreachable();
}

static __inline uint64_t darbyhash_rot64(uint64_t v, unsigned s) {
  return (v >> s) | (v << (64 - s));
}

static __inline uint64_t darbyhash_mix(uint64_t v, uint64_t p) {
  v *= p;
  return v ^ darbyhash_rot64(v, darbyhash_s0);
}


/* xor high and low parts of full 128-bit product */
static __inline uint64_t darbyhash_mux64(uint64_t v, uint64_t p) {
  __uint128_t r = (__uint128_t)v * (__uint128_t)p;
  /* modern GCC could nicely optimize this */
  return r ^ (r >> 64);
}

static uint64_t 
darbyhash(const void *data, size_t len, uint64_t seed) {
  uint64_t a = seed;
  uint64_t b = len;

  if (unlikely(len > 32)) {
    __m128i x = _mm_set_epi64x(a, b);
    __m128i y = _mm_aesenc_si128(x, _mm_set_epi64x(darbyhash_p0, darbyhash_p1));

    const __m128i *v = (const __m128i *)data;
    const __m128i *const detent =
        (const __m128i *)((const uint8_t *)data + (len & ~15ul));
    data = detent;

    if (len & 16) {
      x = _mm_add_epi64(x, _mm_loadu_si128(v++));
      y = _mm_aesenc_si128(x, y);
    }
    len &= 15;

    if (v + 7 < detent) {
      __m128i salt = y;
      do {
        __m128i t = _mm_aesenc_si128(_mm_loadu_si128(v++), salt);
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));

        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));

        salt = _mm_add_epi64(salt, _mm_set_epi64x(darbyhash_p2, darbyhash_p3));
        t = _mm_aesenc_si128(x, t);
        x = _mm_add_epi64(y, x);
        y = t;
      } while (v + 7 < detent);
    }

    while (v < detent) {
      __m128i v0y = _mm_add_epi64(y, _mm_loadu_si128(v++));
      __m128i v1x = _mm_sub_epi64(x, _mm_loadu_si128(v++));
      x = _mm_aesdec_si128(x, v0y);
      y = _mm_aesdec_si128(y, v1x);
    }

    x = _mm_add_epi64(_mm_aesdec_si128(x, _mm_aesenc_si128(y, x)), y);
    a = _mm_cvtsi128_si64(x);
    b = _mm_extract_epi64(x, 1);
  }

  const uint64_t *v = (const uint64_t *)data;
  switch (len) {
  default:
    b += darbyhash_mux64(*v++, darbyhash_p4);
  case 24:
  case 23:
  case 22:
  case 21:
  case 20:
  case 19:
  case 18:
  case 17:
    a += darbyhash_mux64(*v++, darbyhash_p3);
  case 16:
  case 15:
  case 14:
  case 13:
  case 12:
  case 11:
  case 10:
  case 9:
    b += darbyhash_mux64(*v++, darbyhash_p2);
  case 8:
  case 7:
  case 6:
  case 5:
  case 4:
  case 3:
  case 2:
  case 1:
    a += darbyhash_mux64(darbyhash_tail64_le(v, len), darbyhash_p1);
  case 0:
    return darbyhash_mux64(darbyhash_rot64(a + b, darbyhash_s1), darbyhash_p4) + darbyhash_mix(a ^ b, darbyhash_p0);
  }
}

static uint64_t 
darbyhash_noavx(const void *data, size_t len, uint64_t seed) {
  uint64_t a = seed;
  uint64_t b = len;

  if (unlikely(len > 32)) {
    __m128i x = _mm_set_epi64x(a, b);
    __m128i y = _mm_aesenc_si128(x, _mm_set_epi64x(darbyhash_p0, darbyhash_p1));

    const __m128i *v = (const __m128i *)data;
    const __m128i *const detent =
        (const __m128i *)((const uint8_t *)data + (len & ~15ul));
    data = detent;

    if (len & 16) {
      x = _mm_add_epi64(x, _mm_loadu_si128(v++));
      y = _mm_aesenc_si128(x, y);
    }
    len &= 15;

    if (v + 7 < detent) {
      __m128i salt = y;
      do {
        __m128i t = _mm_aesenc_si128(_mm_loadu_si128(v++), salt);
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));

        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v++));

        salt = _mm_add_epi64(salt, _mm_set_epi64x(darbyhash_p2, darbyhash_p3));
        t = _mm_aesenc_si128(x, t);
        x = _mm_add_epi64(y, x);
        y = t;
      } while (v + 7 < detent);
    }

    while (v < detent) {
      __m128i v0y = _mm_add_epi64(y, _mm_loadu_si128(v++));
      __m128i v1x = _mm_sub_epi64(x, _mm_loadu_si128(v++));
      x = _mm_aesdec_si128(x, v0y);
      y = _mm_aesdec_si128(y, v1x);
    }

    x = _mm_add_epi64(_mm_aesdec_si128(x, _mm_aesenc_si128(y, x)), y);
#if defined(__x86_64__) || defined(_M_X64)
    a = _mm_cvtsi128_si64(x);
#if defined(__SSE4_1__)
    b = _mm_extract_epi64(x, 1);
#else
    b = _mm_cvtsi128_si64(_mm_unpackhi_epi64(x, x));
#endif
#else
    a = (uint32_t)_mm_cvtsi128_si32(x);
#if defined(__SSE4_1__)
    a |= (uint64_t)_mm_extract_epi32(x, 1) << 32;
    b = (uint32_t)_mm_extract_epi32(x, 2) |
        (uint64_t)_mm_extract_epi32(x, 3) << 32;
#else
    a |= (uint64_t)_mm_cvtsi128_si32(_mm_shuffle_epi32(x, 1)) << 32;
    x = _mm_unpackhi_epi64(x, x);
    b = (uint32_t)_mm_cvtsi128_si32(x);
    b |= (uint64_t)_mm_cvtsi128_si32(_mm_shuffle_epi32(x, 1)) << 32;
#endif
#endif
  }

  const uint64_t *v = (const uint64_t *)data;
  switch (len) {
  default:
    b += darbyhash_mux64(*v++, darbyhash_p4);
  case 24:
  case 23:
  case 22:
  case 21:
  case 20:
  case 19:
  case 18:
  case 17:
    a += darbyhash_mux64(*v++, darbyhash_p3);
  case 16:
  case 15:
  case 14:
  case 13:
  case 12:
  case 11:
  case 10:
  case 9:
    b += darbyhash_mux64(*v++, darbyhash_p2);
  case 8:
  case 7:
  case 6:
  case 5:
  case 4:
  case 3:
  case 2:
  case 1:
    a += darbyhash_mux64(darbyhash_tail64_le(v, len), darbyhash_p1);
  case 0:
    return darbyhash_mux64(darbyhash_rot64(a + b, darbyhash_s1), darbyhash_p4) + darbyhash_mix(a ^ b, darbyhash_p0);
  }
}

static uint64_t
darbyhash_baseline(const void *data, size_t len, uint64_t seed) {
  uint64_t a = seed;
  uint64_t b = len;

  const int need_align = (((uintptr_t)data) & 7) != 0 && !UNALIGNED_OK;
  uint64_t align[4];

  if (unlikely(len > 32)) {
    uint64_t c = darbyhash_rot64(len, darbyhash_s1) + seed;
    uint64_t d = len ^ darbyhash_rot64(seed, darbyhash_s1);
    const void *detent = (const uint8_t *)data + len - 31;
    do {
      const uint64_t *v = (const uint64_t *)data;
      if (unlikely(need_align))
        v = (const uint64_t *)memcpy(&align, v, 32);

      uint64_t w0 = darbyhash_fetch64_le(v + 0);
      uint64_t w1 = darbyhash_fetch64_le(v + 1);
      uint64_t w2 = darbyhash_fetch64_le(v + 2);
      uint64_t w3 = darbyhash_fetch64_le(v + 3);

      uint64_t d02 = w0 ^ darbyhash_rot64(w2 + d, darbyhash_s1);
      uint64_t c13 = w1 ^ darbyhash_rot64(w3 + c, darbyhash_s1);
      c += a ^ darbyhash_rot64(w0, darbyhash_s0);
      d -= b ^ darbyhash_rot64(w1, darbyhash_s2);
      a ^= darbyhash_p1 * (d02 + w3);
      b ^= darbyhash_p0 * (c13 + w2);
      data = (const uint64_t *)data + 4;
    } while (likely(data < detent));

    a ^= darbyhash_p6 * (darbyhash_rot64(c, darbyhash_s1) + d);
    b ^= darbyhash_p5 * (c + darbyhash_rot64(d, darbyhash_s1));
    len &= 31;
  }

  const uint64_t *v = (const uint64_t *)data;
  if (unlikely(need_align) && len > 8)
    v = (const uint64_t *)memcpy(&align, v, len);

  switch (len) {
  default:
    b += darbyhash_mux64(darbyhash_fetch64_le(v++), darbyhash_p4);
  case 24:
  case 23:
  case 22:
  case 21:
  case 20:
  case 19:
  case 18:
  case 17:
    a += darbyhash_mux64(darbyhash_fetch64_le(v++), darbyhash_p3);
  case 16:
  case 15:
  case 14:
  case 13:
  case 12:
  case 11:
  case 10:
  case 9:
    b += darbyhash_mux64(darbyhash_fetch64_le(v++), darbyhash_p2);
  case 8:
  case 7:
  case 6:
  case 5:
  case 4:
  case 3:
  case 2:
  case 1:
    a += darbyhash_mux64(darbyhash_tail64_le(v, len), darbyhash_p1);
  case 0:
    return darbyhash_mux64(darbyhash_rot64(a + b, darbyhash_s1), darbyhash_p4) + darbyhash_mix(a ^ b, darbyhash_p0);
  }
}

static uint64_t
darbyhash_512(const void *data, size_t len, uint64_t seed) {
  uint64_t a = seed;
  uint64_t b = len;

  const int need_align = (((uintptr_t)data) & 7) != 0 && !UNALIGNED_OK;
  uint64_t align[4];
  const uint64_t *v;
  if (unlikely(len > 32)) {
    __m128i x = _mm_set_epi64x(a, b);
    __m128i y = _mm_aesenc_si128(x, _mm_set_epi64x(darbyhash_p0, darbyhash_p1));

    const __m128i *v1 = (const __m128i *)data;
    const __m128i *const detent =
        (const __m128i *)((const uint8_t *)data + (len & ~15ul));
    data = detent;

    if (len & 16) {
      x = _mm_add_epi64(x, _mm_loadu_si128(v1++));
      y = _mm_aesenc_si128(x, y);
    }
    len &= 15;

    if (v1 + 7 < detent) {
      __m128i salt = y;
      do {
        __m128i t = _mm_aesenc_si128(_mm_loadu_si128(v1++), salt);
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));

        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));

        salt = _mm_add_epi64(salt, _mm_set_epi64x(darbyhash_p2, darbyhash_p3));
        t = _mm_aesenc_si128(x, t);
        x = _mm_add_epi64(y, x);
        y = t;
      } while (v1 + 7 < detent);
    }

    while (v1 < detent) {
      __m128i v0y = _mm_add_epi64(y, _mm_loadu_si128(v1++));
      __m128i v1x = _mm_sub_epi64(x, _mm_loadu_si128(v1++));
      x = _mm_aesdec_si128(x, v0y);
      y = _mm_aesdec_si128(y, v1x);
    }

    x = _mm_add_epi64(_mm_aesdec_si128(x, _mm_aesenc_si128(y, x)), y);
    a = _mm_cvtsi128_si64(x);
    b = _mm_extract_epi64(x, 1);

    v = (const uint64_t *)data;
  }
  else {
      v = (const uint64_t *)data;
      if (unlikely(need_align) && len > 8)
        v = (const uint64_t *)memcpy(&align, v, len);
  }

  switch (len) {
  default:
    b += darbyhash_mux64(darbyhash_fetch64_le(v++), darbyhash_p4);
  case 24:
  case 23:
  case 22:
  case 21:
  case 20:
  case 19:
  case 18:
  case 17:
    a += darbyhash_mux64(darbyhash_fetch64_le(v++), darbyhash_p3);
  case 16:
  case 15:
  case 14:
  case 13:
  case 12:
  case 11:
  case 10:
  case 9:
    b += darbyhash_mux64(darbyhash_fetch64_le(v++), darbyhash_p2);
  case 8:
  case 7:
  case 6:
  case 5:
  case 4:
  case 3:
  case 2:
  case 1:
    a += darbyhash_mux64(darbyhash_tail64_le(v, len), darbyhash_p1);
  case 0:
    return darbyhash_mux64(darbyhash_rot64(a + b, darbyhash_s1), darbyhash_p4) + darbyhash_mix(a ^ b, darbyhash_p0);
  }
}

static uint64_t
darbyhash_boring(const void *data, size_t len, uint64_t seed) {
  uint64_t a = seed;
  uint64_t b = len;

  const int need_align = (((uintptr_t)data) & 7) != 0 && !UNALIGNED_OK;
  uint64_t align[4];
  const uint64_t *v;
  if (unlikely(len > 32)) {
    __m128i x = _mm_set_epi64x(a, b);
    __m128i y = _mm_aesenc_si128(x, _mm_set_epi64x(darbyhash_p0, darbyhash_p1));

    const __m128i *v1 = (const __m128i *)data;
    const __m128i *const detent =
        (const __m128i *)((const uint8_t *)data + (len & ~15ul));
    data = detent;

    if (len & 16) {
      x = _mm_add_epi64(x, _mm_loadu_si128(v1++));
      y = _mm_aesenc_si128(x, y);
    }
    len &= 15;

    if (v1 + 7 < detent) {
      __m128i salt = y;
      do {
        __m128i t = _mm_aesenc_si128(_mm_loadu_si128(v1++), salt);
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));

        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));
        t = _mm_aesdec_si128(t, _mm_loadu_si128(v1++));

        salt = _mm_add_epi64(salt, _mm_set_epi64x(darbyhash_p2, darbyhash_p3));
        t = _mm_aesenc_si128(x, t);
        x = _mm_add_epi64(y, x);
        y = t;
      } while (v1 + 7 < detent);
    }

    __m512i  * v512 = (__m512i *) v1;
    while (v1 < detent) {
      __m512i v0y = _mm512_add_epi64(y, v512++);
      __m512i v1x = _mm_sub_epi64(x, v512++));
      x = _mm_aesdec_si128(x, v0y);
      y = _mm_aesdec_si128(y, v1x);
    }

    x = _mm_add_epi64(_mm_aesdec_si128(x, _mm_aesenc_si128(y, x)), y);
    a = _mm_cvtsi128_si64(x);
    b = _mm_extract_epi64(x, 1);

    v = (const uint64_t *)data;
  }
  else {
      v = (const uint64_t *)data;
      if (unlikely(need_align) && len > 8)
        v = (const uint64_t *)memcpy(&align, v, len);
  }

  switch (len) {
  default:
    b += darbyhash_mux64(darbyhash_fetch64_le(v++), darbyhash_p4);
  case 24:
  case 23:
  case 22:
  case 21:
  case 20:
  case 19:
  case 18:
  case 17:
    a += darbyhash_mux64(darbyhash_fetch64_le(v++), darbyhash_p3);
  case 16:
  case 15:
  case 14:
  case 13:
  case 12:
  case 11:
  case 10:
  case 9:
    b += darbyhash_mux64(darbyhash_fetch64_le(v++), darbyhash_p2);
  case 8:
  case 7:
  case 6:
  case 5:
  case 4:
  case 3:
  case 2:
  case 1:
    a += darbyhash_mux64(darbyhash_tail64_le(v, len), darbyhash_p1);
  case 0:
    return darbyhash_mux64(darbyhash_rot64(a + b, darbyhash_s1), darbyhash_p4) + darbyhash_mix(a ^ b, darbyhash_p0);
  }
}
