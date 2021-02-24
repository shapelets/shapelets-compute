#include <algos/internal/util.h>

std::string algos::util::khiva_file_path(const std::string &path) {
    auto pos = path.rfind("khiva");
    return (pos == std::string::npos) ? path : path.substr(pos);
}
