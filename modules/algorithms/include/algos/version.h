// Copyright (c) 2019 Shapelets.io
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algos/defines.h>

#include <string>

constexpr char versionMajor[] = "0";
constexpr char versionMinor[] = "5";
constexpr char versionPatch[] = "0";
constexpr char versionSha1[] = "e2970e5";
constexpr char versionShort[] = "0.5.0";
constexpr char buildType[] = "Release";

namespace khiva {

/**
 * @brief Returns the version of Khiva Library.
 *
 * @return std::string A string containing the version number of Khiva library.
 */
KHIVAAPI std::string version();

}  // namespace khiva
