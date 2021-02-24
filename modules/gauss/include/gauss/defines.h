#ifndef GAUSS_HEADER_
#define GAUSS_HEADER_

#if defined(_WIN32) || defined(_MSC_VER)
#if defined(GAUSS_EXPORTS)
#define GAUSSAPI __declspec(dllexport)
#else
#define GAUSSAPI __declspec(dllimport)
#endif
#else
#define GAUSSAPI
#endif

#endif
