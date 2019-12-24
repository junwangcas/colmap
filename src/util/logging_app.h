#pragma once
#include <glog/logging.h>

#define ERROR()               LOG(ERROR)
#define INFO()                LOG(INFO)
#define WARNING()             LOG(WARNING)
#define FATAL()               LOG(FATAL)
#define DFATAL()              LOG(DFATAL)

#define ERROR_IF(condition)   LOG_IF(ERROR, condition)<< #condition" is true: "
#define INFO_IF(condition)    LOG_IF(INFO, condition)<< #condition" is true: "
#define WARNING_IF(condition) LOG_IF(WARNING, condition)<< #condition" is true: "
#define FATAL_IF(condition)   LOG_IF(FATAL, condition)<< #condition" is true: "
#define DFATAL_IF(condition)  LOG_IF(DFATAL, condition)<< #condition" is true: "

#define ERROR_IF_NOT(condition)   LOG_IF(ERROR, !(condition))<< #condition" is false: "
#define INFO_IF_NOT(condition)    LOG_IF(INFO, !(condition))<< #condition" is false: "
#define WARNING_IF_NOT(condition) LOG_IF(WARNING, !(condition))<< #condition" is false: "
#define FATAL_IF_NOT(condition)   LOG_IF(FATAL, !(condition))<< #condition" is false: "
#define DFATAL_IF_NOT(condition)  LOG_IF(DFATAL, !(condition))<< #condition" is false: "

#define DATAINFO(name, value) INFO() << #name": " << value

#define SET_LOG_LEVEL(level) FLAGS_minloglevel=google::GLOG_##level

#define INITIAL_LOG(logdir) google::InitGoogleLogging("./");FLAGS_log_dir=logdir

#define LOG_TO_ERR() FLAGS_logtostderr=1
