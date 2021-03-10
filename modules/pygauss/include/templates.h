#ifndef __TEMPLATES_H__
#define __TEMPLATES_H__


#define UNARY_TEMPLATE_FN_LAMBDA(NAME, OP, HELP)                                    \
    m.def(#NAME,                                                                    \
          [](const py::object &array_like) {                                        \
                auto conv = pygauss::arraylike::as_array(array_like);               \
                if (!conv)                                                          \
                    throw std::runtime_error(#NAME ": Expected array");             \
                return OP(conv.value());                                            \
          },                                                                        \
          py::arg("array_like").none(false),                                        \
          HELP);

#define UNARY_TEMPLATE_FN(NAME, OP, HELP)                                           \
    m.def(#NAME,                                                                    \
          [](const py::object &array_like) {                                        \
                auto conv = pygauss::arraylike::as_array(array_like);               \
                if (!conv)                                                          \
                    throw std::runtime_error(#NAME ": Expected array");             \
                                                                                    \
                 af_array out = nullptr;                                            \
                 throw_on_error(OP(&out, conv.value().get()));                      \
                 return af::array(out);                                             \
          },                                                                        \
          py::arg("array_like").none(false),                                        \
          HELP);


#define BINARY_TEMPLATE_FN(NAME, OP, HELP)                                                                  \
    m.def(#NAME,                                                                                            \
          [](const py::object &left, const py::object &right) {                                             \
              auto result = pygauss::arraylike::as_array(left, right);                                      \
              if (!result)                                                                                  \
                throw std::runtime_error("Operation " #NAME ": Both are scalars");                          \
                                                                                                            \
              auto [l,r] = result.value();                                                                  \
              af_array out = nullptr;                                                                       \
              throw_on_error(OP(&out, l.get(), r.get(), GForStatus::get()));                                \
              return af::array(out);                                                                        \
          },                                                                                                \
          py::arg("left").none(false),                                                                      \
          py::arg("right").none(false),                                                                     \
          HELP);

#define BINARY_TEMPLATE_FN_LAMBDA(NAME, OP, HELP)                                                           \
    m.def(#NAME,                                                                                            \
          [](const py::object &left, const py::object &right) {                                             \
              auto result = pygauss::arraylike::as_array(left, right);                                      \
              if (!result)                                                                                  \
                throw std::runtime_error("Operation " #NAME ": Both are scalars");                          \
                                                                                                            \
              auto [l,r] = result.value();                                                                  \
              return OP(l, r, GForStatus::get());                                                           \
          },                                                                                                \
          py::arg("left").none(false),                                                                      \
          py::arg("right").none(false),                                                                     \
          HELP);

#endif  // __TEMPLATES_H__
