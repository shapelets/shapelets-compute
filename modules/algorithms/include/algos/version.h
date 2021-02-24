#pragma once

#include <algos/defines.h>

#include <string>

constexpr char versionMajor[] = "0";
constexpr char versionMinor[] = "5";
constexpr char versionPatch[] = "0";
constexpr char versionSha1[] = "e2970e5";
constexpr char versionShort[] = "0.5.0";
constexpr char buildType[] = "Release";

namespace algos {

/**
 * @brief Returns the version of Khiva Library.
 *
 * @return std::string A string containing the version number of Khiva library.
 */
ALGOSAPI std::string version();

}  // namespace algos
