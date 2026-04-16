#pragma once
#include <string>

// Thread local variable that holds the most recent exception message with
// backtrace.
extern thread_local std::string torch_exception_what;

// Thread local variable that holds the most recent exception message.
extern thread_local std::string torch_exception_what_without_backtrace;
