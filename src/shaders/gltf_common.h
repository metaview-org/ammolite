#ifndef COMMON_H
#define COMMON_H

#define PI 3.1415926535897932384626433832795
#define PI_2 1.57079632679489661923
#define PI_4 0.785398163397448309616

#define PROJECT(vector4) (vector4.w == 0 ? vector4.xyz : (vector4.xyz / vector4.w))
#define GRAM_SCHMIDT(a, b) (a - (b) * dot((a), (b)))
#define LERP(a, b, t) ((t) * (a) + (1.0 - (t)) * (b))

#endif
