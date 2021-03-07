#ifndef __TEMPLATES_H__
#define __TEMPLATES_H__


#define UNARY_TEMPLATE_FN_LAMBDA(NAME, OP, HELP)                                    \
    m.def(#NAME,                                                                    \
          [](const py::object &array_like,                                          \
             const std::optional<af::dim4> &shape,                                  \
             const std::optional<af::dtype> &dtype) {                               \
                 auto arr = pygauss::arraylike::cast(array_like, false, shape, dtype); \
                 return OP(arr);                                                    \
          },                                                                        \
          py::arg("array_like").none(false),                                        \
          py::arg("shape") = std::nullopt,                                          \
          py::arg("dtype") = std::nullopt,                                          \
          HELP);

#define UNARY_TEMPLATE_FN(NAME, OP, HELP)                                           \
    m.def(#NAME,                                                                    \
          [](const py::object &array_like,                                          \
             const std::optional<af::dim4> &shape,                                  \
             const std::optional<af::dtype> &dtype) {                               \
                 auto arr = pygauss::arraylike::cast(array_like, false, shape, dtype); \
                 af_array out = nullptr;                                            \
                 throw_on_error(OP(&out, arr.get()));                               \
                 return af::array(out);                                             \
          },                                                                        \
          py::arg("array_like").none(false),                                        \
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
              auto isLeftScalar = pygauss::arraylike::is_scalar(left);                                      \
              auto isRightScalar = pygauss::arraylike::is_scalar(right);                                    \
              if (isLeftScalar && isRightScalar) {                                                          \
                  if (shape.has_value()) {                                                                  \
                      l = pygauss::arraylike::try_cast(left, shape, dtype);                                 \
                      r = pygauss::arraylike::try_cast(right, shape, dtype);                                \
                  }                                                                                         \
                  else {                                                                                    \
                      throw std::runtime_error("Operation " #NAME ": Both "                                 \
                      "parameters were scalars a no shape was given.");                                     \
                  }                                                                                         \
              } else if (isLeftScalar) {                                                                    \
                  r = pygauss::arraylike::try_cast(right, shape, dtype);                                    \
                  l = pygauss::arraylike::try_cast(left, r->dims(), r->type());                             \
              } else if (isRightScalar){                                                                    \
                  l = pygauss::arraylike::try_cast(left, shape, dtype);                                     \
                  r = pygauss::arraylike::try_cast(right, l->dims(), l->type());                            \
              }                                                                                             \
              else {                                                                                        \
                  l = pygauss::arraylike::try_cast(left, shape, dtype);                                     \
                  r = pygauss::arraylike::try_cast(right, shape, dtype);                                    \
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
              auto isLeftScalar = pygauss::arraylike::is_scalar(left);                                      \
              auto isRightScalar = pygauss::arraylike::is_scalar(right);                                    \
              if (isLeftScalar && isRightScalar) {                                                          \
                  if (shape.has_value()) {                                                                  \
                      l = pygauss::arraylike::try_cast(left, shape, dtype);                                 \
                      r = pygauss::arraylike::try_cast(right, shape, dtype);                                \
                  }                                                                                         \
                  else {                                                                                    \
                      throw std::runtime_error("Operation " #NAME ": Both "                                 \
                      "parameters were scalars a no shape was given.");                                     \
                  }                                                                                         \
              } else if (isLeftScalar) {                                                                    \
                  r = pygauss::arraylike::try_cast(right, shape, dtype);                                    \
                  l = pygauss::arraylike::try_cast(left, r->dims(), r->type());                             \
              } else if (isRightScalar){                                                                    \
                  l = pygauss::arraylike::try_cast(left, shape, dtype);                                     \
                  r = pygauss::arraylike::try_cast(right, l->dims(), l->type());                            \
              }                                                                                             \
              else {                                                                                        \
                  l = pygauss::arraylike::try_cast(left, shape, dtype);                                     \
                  r = pygauss::arraylike::try_cast(right, shape, dtype);                                    \
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
          py::arg("shape") = std::nullopt,                                                                  \
          py::arg("dtype") = std::nullopt,                                                                  \
          HELP);

#endif  // __TEMPLATES_H__
