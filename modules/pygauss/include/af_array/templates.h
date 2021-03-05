#ifndef AF_ARRAY_TEMPLATES_H
#define AF_ARRAY_TEMPLATES_H


#define UNARY_TEMPLATE_FN_LAMBDA(NAME, OP, HELP)                                    \
    m.def(#NAME,                                                                    \
          [](const py::object &array_like,                                          \
             const std::optional<af::dim4> &shape,                                  \
             const std::optional<af::dtype> &dtype) {                               \
                 auto arr = array_like::to_array(array_like, shape, dtype);         \
                 std::optional<af::array> result = std::nullopt;                    \
                 if (arr.has_value())                                               \
                    result = OP(arr.value());                                       \
                 return result;                                                     \
          },                                                                        \
          py::arg("array_like").none(false),                                        \
          py::kw_only(),                                                            \
          py::arg("shape") = std::nullopt,                                          \
          py::arg("dtype") = std::nullopt,                                          \
          HELP);

#define UNARY_TEMPLATE_FN(NAME, OP, HELP)                                           \
    m.def(#NAME,                                                                    \
          [](const py::object &array_like,                                          \
             const std::optional<af::dim4> &shape,                                  \
             const std::optional<af::dtype> &dtype) {                               \
                 auto arr = array_like::to_array(array_like, shape, dtype);         \
                 std::optional<af::array> result = std::nullopt;                    \
                 if (arr.has_value()) {                                             \
                    af_array out = nullptr;                                         \
                    throw_on_error(OP(&out, arr->get()));                           \
                    result = af::array(out);                                        \
                 }                                                                  \
                 return result;                                                     \
          },                                                                        \
          py::arg("array_like").none(false),                                        \
          py::kw_only(),                                                            \
          py::arg("shape") = std::nullopt,                                          \
          py::arg("dtype") = std::nullopt,                                          \
          HELP);


#define BINARY_TEMPLATE_FN(NAME, OP, HELP)                                                                  \
    m.def(#NAME,                                                                                            \
          [](const py::object &left,                                                                        \
             const py::object &right,                                                                       \
             const std::optional<af::dim4> &shape,                                                          \
             const std::optional<af::dtype> &dtype) {                                                       \
                                                                                                            \
              af_array out = nullptr;                                                                       \
              std::optional<af::array> l;                                                                   \
              std::optional<af::array> r;                                                                   \
                                                                                                            \
              auto isLeftScalar = array_like::is_scalar(left);                                              \
              auto isRightScalar = array_like::is_scalar(right);                                            \
              if (isLeftScalar && isRightScalar) {                                                          \
                  if (shape.has_value()) {                                                                  \
                      l = array_like::to_array(left, shape, dtype);                                         \
                      r = array_like::to_array(right, shape, dtype);                                        \
                  }                                                                                         \
                  else {                                                                                    \
                      throw std::runtime_error("Operation " #NAME ": Both "                                 \
                      "parameters were scalars a no shape was given.");                                     \
                  }                                                                                         \
              } else if (isLeftScalar) {                                                                    \
                  r = array_like::to_array(right, shape, dtype);                                            \
                  l = array_like::to_array(left, r->dims(), r->type());                                     \
              } else if (isRightScalar){                                                                    \
                  l = array_like::to_array(left, shape, dtype);                                             \
                  r = array_like::to_array(right, l->dims(), l->type());                                    \
              }                                                                                             \
              else {                                                                                        \
                  l = array_like::to_array(left, shape, dtype);                                             \
                  r = array_like::to_array(right, shape, dtype);                                            \
              }                                                                                             \
                                                                                                            \
              if (!l.has_value() || !r.has_value()) {                                                       \
                  std::ostringstream stm;                                                                   \
                  stm << "Operation " << #NAME << ": ";                                                     \
                  if (!l.has_value())                                                                       \
                      stm << "Left operand " << py::repr(left) << " couldn't be interpreted as an array";   \
                  if (!r.has_value())                                                                       \
                      stm << "Right operand " << py::repr(right) << " couldn't be interpreted as an array"; \
                                                                                                            \
                  auto err_msg = stm.str();                                                                 \
                  spd::error(err_msg);                                                                      \
                  throw std::runtime_error(err_msg);                                                        \
              }                                                                                             \
                                                                                                            \
              throw_on_error(OP(&out, l->get(), r->get(), GForStatus::get()));                              \
              return af::array(out);                                                                        \
          },                                                                                                \
          py::arg("left").none(false),                                                                      \
          py::arg("right").none(false),                                                                     \
          py::kw_only(),                                                                                    \
          py::arg("shape") = std::nullopt,                                                                  \
          py::arg("dtype") = std::nullopt,                                                                  \
          HELP);

#define BINARY_TEMPLATE_FN_LAMBDA(NAME, OP, HELP)                                                           \
    m.def(#NAME,                                                                                            \
          [](const py::object &left,                                                                        \
             const py::object &right,                                                                       \
             const std::optional<af::dim4> &shape,                                                          \
             const std::optional<af::dtype> &dtype) {                                                       \
                                                                                                            \
              af_array out = nullptr;                                                                       \
              std::optional<af::array> l;                                                                   \
              std::optional<af::array> r;                                                                   \
                                                                                                            \
              auto isLeftScalar = array_like::is_scalar(left);                                              \
              auto isRightScalar = array_like::is_scalar(right);                                            \
              if (isLeftScalar && isRightScalar) {                                                          \
                  if (shape.has_value()) {                                                                  \
                      l = array_like::to_array(left, shape, dtype);                                         \
                      r = array_like::to_array(right, shape, dtype);                                        \
                  }                                                                                         \
                  else {                                                                                    \
                      throw std::runtime_error("Operation " #NAME ": Both "                                 \
                      "parameters were scalars a no shape was given.");                                     \
                  }                                                                                         \
              } else if (isLeftScalar) {                                                                    \
                  r = array_like::to_array(right, shape, dtype);                                            \
                  l = array_like::to_array(left, r->dims(), r->type());                                     \
              } else if (isRightScalar){                                                                    \
                  l = array_like::to_array(left, shape, dtype);                                             \
                  r = array_like::to_array(right, l->dims(), l->type());                                    \
              }                                                                                             \
              else {                                                                                        \
                  l = array_like::to_array(left, shape, dtype);                                             \
                  r = array_like::to_array(right, shape, dtype);                                            \
              }                                                                                             \
                                                                                                            \
              if (!l.has_value() || !r.has_value()) {                                                       \
                  std::ostringstream stm;                                                                   \
                  stm << "Operation " << #NAME << ": ";                                                     \
                  if (!l.has_value())                                                                       \
                      stm << "Left operand " << py::repr(left) << " couldn't be interpreted as an array";   \
                  if (!r.has_value())                                                                       \
                      stm << "Right operand " << py::repr(right) << " couldn't be interpreted as an array"; \
                                                                                                            \
                  auto err_msg = stm.str();                                                                 \
                  spd::error(err_msg);                                                                      \
                  throw std::runtime_error(err_msg);                                                        \
              }                                                                                             \
                                                                                                            \
              return OP(l.value(), r.value(), GForStatus::get());                                           \
          },                                                                                                \
          py::arg("left").none(false),                                                                      \
          py::arg("right").none(false),                                                                     \
          py::kw_only(),                                                                                    \
          py::arg("shape") = std::nullopt,                                                                  \
          py::arg("dtype") = std::nullopt,                                                                  \
          HELP);


#endif //AF_ARRAY_TEMPLATES_H
