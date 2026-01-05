#pragma once

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>
#include <source_location>

[[maybe_unused]] static int debugPoint(int line) {
    return line; // breakpoint can be set in this line to debug assertion raises
}

inline void rassert_print(std::ostream &stream) {}

template <typename First, typename... Rest>
inline void rassert_print(std::ostream &stream, const First &first, const Rest &...rest) {
    stream << first;
    if (sizeof...(rest) > 0)
        stream << " ";
    rassert_print(stream, rest...);
}

template <typename Exception, typename ErrorCode, typename... Args>
[[noreturn]] inline void rassert_throw(const ErrorCode &error_code, int line, const Args &...args) {
    std::stringstream ss;
    ss << "Assertion \"";
    rassert_print(ss, error_code, args...);
    ss << "\" failed at line " << debugPoint(line);
    std::stringstream ec;
    ec << error_code;
    throw Exception(ss.str(), ec.str());
}

// common macros
#define rassert_ex(condition, exception_type, error_code, ...)                                                         \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            rassert_throw<exception_type>((error_code), __LINE__, ##__VA_ARGS__);                                      \
        }                                                                                                              \
    } while (0)

class assertion_error : public std::runtime_error {
  public:
    assertion_error(const std::string &message, const std::string &code = std::string())
        : std::runtime_error(message), code_(code) {}
    assertion_error(const char *message, const std::string &code = std::string())
        : std::runtime_error(message), code_(code) {}

    const std::string &code() const { return code_; }

  protected:
    std::string code_;
};

inline std::string format_code_location(std::source_location loc) {
    return "\n" + std::string(loc.file_name()) + ":" + std::to_string(loc.line()) + ":" + std::to_string(loc.column());
}

#define rassert(condition, error_code, ...) rassert_ex((condition), assertion_error, (error_code), ##__VA_ARGS__)

// Usage:
// rassert(<condition>, <pseudo unique code>); // unique code is helpful to find with Ctrl+F the failed line
// rassert(p.x() >= 0 && p.y() >= 0, 85974627394081);
// rassert(p.x() >= 0 && p.y() >= 0, 9784057812341, p.x(), p.y());
// rassert(p.x() >= 0 && p.y() >= 0, 7894732123523, "Wrong position", p.x(), p.y());
// rassert(p.x() >= 0 && p.y() >= 0, "Wrong position", p.x(), p.y());
// rassert(p.x() >= 0 && p.y() >= 0, "Wrong position", p.x(), p.y(), format_location(loc));
