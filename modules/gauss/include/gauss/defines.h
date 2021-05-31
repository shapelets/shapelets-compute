/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

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
