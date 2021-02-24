#ifndef _ALGOS_HEADER_
#define _ALGOS_HEADER_

#if defined(_WIN32) || defined(_MSC_VER)
#if defined(ALGO_EXPORTS)
#define ALGOSAPI __declspec(dllexport)
#else
#define ALGOSAPI __declspec(dllimport)
#endif
#else
#define ALGOSAPI
#endif

#endif
